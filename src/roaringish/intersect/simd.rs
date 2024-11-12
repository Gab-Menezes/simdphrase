#[allow(unused_imports)]
use std::{
    mem::MaybeUninit,
    simd::{cmp::SimdPartialOrd, Simd},
    arch::x86_64::__m512i
};

use crate::roaringish::{BorrowRoaringishPacked, Aligned64};

use super::naive::NaiveIntersect;
use super::{private::IntersectSeal, Intersect};
use crate::roaringish::Aligned;

const N: usize = 8;

#[cfg(target_feature = "avx512vp2intersect")]
#[inline(always)]
unsafe fn vp2intersectq(a: __m512i, b: __m512i) -> (u8, u8) {
    use std::arch::{asm, x86_64::__mmask8};

    let mut mask0: __mmask8;
    let mut mask1: __mmask8;
    asm!(
        "vp2intersectq k2, {0}, {1}",
        in(zmm_reg) a,
        in(zmm_reg) b,
        out("k2") mask0,
        out("k3") mask1,
        options(pure, nomem, nostack),
    );

    (mask0, mask1)
}

#[cfg(not(target_feature = "avx512vp2intersect"))]
#[inline(always)]
unsafe fn vp2intersectq(a: __m512i, b: __m512i) -> (u8, u8) {
    use std::arch::x86_64::{
        _mm512_alignr_epi32, _mm512_cmpeq_epi64_mask, _mm512_shuffle_epi32, _MM_PERM_BADC,
    };

    let a1 = _mm512_alignr_epi32(a, a, 4);
    let a2 = _mm512_alignr_epi32(a, a, 8);
    let a3 = _mm512_alignr_epi32(a, a, 12);

    let b1 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);

    let m00 = _mm512_cmpeq_epi64_mask(a, b);
    let m01 = _mm512_cmpeq_epi64_mask(a, b1);
    let m10 = _mm512_cmpeq_epi64_mask(a1, b);
    let m11 = _mm512_cmpeq_epi64_mask(a1, b1);
    let m20 = _mm512_cmpeq_epi64_mask(a2, b);
    let m21 = _mm512_cmpeq_epi64_mask(a2, b1);
    let m30 = _mm512_cmpeq_epi64_mask(a3, b);
    let m31 = _mm512_cmpeq_epi64_mask(a3, b1);

    let mask0 = m00
        | m01
        | (m10 | m11).rotate_left(2)
        | (m20 | m21).rotate_left(4)
        | (m30 | m31).rotate_left(6);

    let m0 = m00 | m10 | m20 | m30;
    let m1 = m01 | m11 | m21 | m31;
    let mask1 = m0 | ((0x55 & m1) << 1) | ((m1 >> 1) & 0x55);

    (mask0, mask1)
}

#[inline(always)]
unsafe fn analyze_msb(
    va: __m512i,
    a_values: &[u16],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    msb_values_result: &mut [MaybeUninit<u16>],
    j: &mut usize,
    msb_mask: Simd::<u16, N>
) {
    use std::arch::x86_64::{_mm512_mask_compressstoreu_epi64, _mm_mask_compressstoreu_epi16, _mm_loadu_epi16};

    let values: Simd<u16, N> =
        unsafe { _mm_loadu_epi16(a_values.as_ptr().add(lhs_i) as *const _).into() };
    let vmsb_set = (values & msb_mask).simd_gt(Simd::splat(0));

    let mask = vmsb_set.to_bitmask() as u8;
    let va: Simd<u64, N> = va.into();
    let doc_id_groups_plus_one: Simd<u64, N> = Simd::splat(1) + va;
    unsafe {
        // TODO: avoid compressstore on zen4
        _mm512_mask_compressstoreu_epi64(
            msb_doc_id_groups_result.as_mut_ptr().add(*j) as *mut _,
            mask,
            doc_id_groups_plus_one.into(),
        );
        _mm_mask_compressstoreu_epi16(
            msb_values_result.as_mut_ptr().add(*j) as *mut _,
            mask,
            values.into(),
        );
    }
    *j += mask.count_ones() as usize;
}

#[inline(always)]
fn rotl_u16(a: Simd<u16, N>, i: u16) -> Simd<u16, N> {
    const M: u16 = 16;
    let i = i % M;
    let p0 = a << i;
    let p1 = a >> (M - i);
    p0 | p1
}

pub struct SimdIntersect;
impl IntersectSeal for SimdIntersect {}

impl Intersect for SimdIntersect {
    #[inline(always)]
    fn inner_intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        msb_values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        j: &mut usize,

        lhs_len: u16,
        msb_mask: u16,
        lsb_mask: u16
    ) {
        use std::arch::x86_64::{
            _mm512_load_epi64, _mm512_mask_compressstoreu_epi64, _mm_load_si128,
            _mm_maskz_compress_epi16, _mm_storeu_epi16,
        };

        let simd_msb_mask = Simd::splat(msb_mask);
        let simd_lsb_mask = Simd::splat(lsb_mask);

        let end_lhs = lhs.doc_id_groups.len() / N * N;
        let end_rhs = rhs.doc_id_groups.len() / N * N;
        let a = unsafe { lhs.doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs.doc_id_groups.get_unchecked(..end_rhs) };
        let a_values = unsafe { lhs.values.get_unchecked(..end_lhs) };
        let b_values = unsafe { rhs.values.get_unchecked(..end_rhs) };
        let mut need_to_analyze_msb = false;
        assert_eq!(a.len() % N, 0);
        assert_eq!(b.len() % N, 0);
        assert_eq!(a_values.len() % N, 0);
        assert_eq!(b_values.len() % N, 0);

        while *lhs_i < a.len() && *rhs_i < b.len() {
            let (va, vb) = unsafe {
                let va = _mm512_load_epi64(a.as_ptr().add(*lhs_i) as *const _);
                let vb = _mm512_load_epi64(b.as_ptr().add(*rhs_i) as *const _);
                (va, vb)
            };

            let (mask_a, mask_b) = unsafe { vp2intersectq(va, vb) };

            unsafe {
                // TODO: avoid compressstore on zen4
                _mm512_mask_compressstoreu_epi64(
                    doc_id_groups_result.as_mut_ptr().add(*i) as *mut _,
                    mask_a,
                    va,
                );

                let va_values: Simd<u16, N> = _mm_maskz_compress_epi16(
                    mask_a,
                    _mm_load_si128(a_values.as_ptr().add(*lhs_i) as *const _),
                )
                .into();
                let vb_values: Simd<u16, N> = _mm_maskz_compress_epi16(
                    mask_b,
                    _mm_load_si128(b_values.as_ptr().add(*rhs_i) as *const _),
                )
                .into();

                if FIRST {
                    _mm_storeu_epi16(
                        values_result.as_mut_ptr().add(*i) as *mut _,
                        ((va_values << lhs_len) & vb_values).into(),
                    );
                } else {
                    _mm_storeu_epi16(
                        values_result.as_mut_ptr().add(*i) as *mut _,
                        (rotl_u16(va_values, lhs_len) & simd_lsb_mask & vb_values).into(),
                    );
                }

                *i += mask_a.count_ones() as usize;
            }

            let last_a = unsafe { *a.get_unchecked(*lhs_i + N - 1) };
            let last_b = unsafe { *b.get_unchecked(*rhs_i + N - 1) };

            if FIRST {
                if last_a <= last_b {
                    unsafe {
                        analyze_msb(va, a_values, *lhs_i, msb_doc_id_groups_result, msb_values_result, j, simd_msb_mask);
                    }
                    *lhs_i += N;
                }
            } else {
                *lhs_i += N * (last_a <= last_b) as usize;
            }
            *rhs_i += N * (last_b <= last_a) as usize;
            need_to_analyze_msb = last_b < last_a;
        }

        if FIRST
            && need_to_analyze_msb
            && !(*lhs_i < lhs.doc_id_groups.len() && *rhs_i < rhs.doc_id_groups.len())
        {
            unsafe {
                let va = _mm512_load_epi64(a.as_ptr().add(*lhs_i) as *const _);
                analyze_msb(va, a_values, *lhs_i, msb_doc_id_groups_result, msb_values_result, j, simd_msb_mask);
            };
        }

        NaiveIntersect::inner_intersect::<FIRST>(
            lhs,
            rhs,
            lhs_i,
            rhs_i,

            doc_id_groups_result,
            values_result,
            i,

            msb_doc_id_groups_result,
            msb_values_result,
            j,

            lhs_len,
            msb_mask,
            lsb_mask
        );
    }

    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize {
        lhs.doc_id_groups.len().min(rhs.doc_id_groups.len()) + 1 + N
    }
}
