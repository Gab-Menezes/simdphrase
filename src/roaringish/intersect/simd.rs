#[allow(unused_imports)]
use std::arch::x86_64::{__m256i, __m512i};
#[allow(unused_imports)]
use std::{
    intrinsics::assume,
    mem::MaybeUninit,
    simd::{cmp::SimdPartialOrd, Simd},
};

use crate::roaringish::{BorrowRoaringishPacked, Aligned64};

use super::naive::NaiveIntersect;
use super::{private::IntersectSeal, Intersect};

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
))]
const N: usize = 8;
#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
)))]
const N: usize = 4;

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
    target_feature = "avx512vp2intersect",
))]
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

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
    not(target_feature = "avx512vp2intersect"),
))]
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

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
)))]
#[inline(always)]
unsafe fn vp2intersectq(a: __m256i, b: __m256i) -> (Simd<u64, N>, Simd<u64, N>) {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi64, _mm256_or_si256, _mm256_permute2x128_si256, _mm256_shuffle_epi32,
        _MM_PERM_BADC,
    };

    let a1 = _mm256_permute2x128_si256(a, a, 1);
    let b1 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);

    let m00 = _mm256_cmpeq_epi64(a, b);
    let m01 = _mm256_cmpeq_epi64(a, b1);
    let m10 = _mm256_cmpeq_epi64(a1, b);
    let m11 = _mm256_cmpeq_epi64(a1, b1);

    let l = _mm256_or_si256(m00, m01);
    let h = _mm256_or_si256(m10, m11);
    let mask0 = _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
    // let mask0 = unsafe { std::mem::transmute::<__m256i, Mask<i64, 4>>(mask0).to_bitmask() as u8 };

    let l = _mm256_or_si256(m00, m10);
    let h = _mm256_or_si256(m01, m11);
    let mask1 = _mm256_or_si256(l, _mm256_shuffle_epi32(h, _MM_PERM_BADC));
    // let mask1 = unsafe { std::mem::transmute::<__m256i, Mask<i64, 4>>(mask1).to_bitmask() as u8 };
    // (mask0, mask1)
    (mask0.into(), mask1.into())
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
))]
#[inline(always)]
unsafe fn analyze_msb(
    va: __m512i,
    a_values: &[u16],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) {
    use std::arch::x86_64::{_mm512_mask_compressstoreu_epi64, _mm_loadu_epi16};

    let vvalues: Simd<u16, N> =
        unsafe { _mm_loadu_epi16(a_values.as_ptr().add(lhs_i) as *const _).into() };
    let vmsb_set = (vvalues & Simd::splat(0x8000)).simd_gt(Simd::splat(0));

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
    }
    *j += mask.count_ones() as usize;
}

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
)))]
#[inline(always)]
unsafe fn analyze_msb(
    a: &[u64],
    a_values: &[u16],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) {
    let values = unsafe { a_values.get_unchecked(lhs_i..(lhs_i + 4)) };

    for (k, v) in values.iter().enumerate() {
        let doc_id_groups = unsafe { *a.get_unchecked(lhs_i + k) };
        unsafe {
            msb_doc_id_groups_result
                .get_unchecked_mut(*j)
                .write(doc_id_groups + 1)
        };
        let set = (*v & 0x8000) > 0;
        *j += set as usize;
    }
}

pub struct SimdIntersect;
impl IntersectSeal for SimdIntersect {}

impl Intersect for SimdIntersect {
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi2",
        target_feature = "avx512dq",
    ))]
    fn inner_intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        j: &mut usize,
    ) {
        use std::arch::x86_64::{
            _mm512_load_epi64, _mm512_mask_compressstoreu_epi64, _mm_load_si128,
            _mm_maskz_compress_epi16, _mm_storeu_epi16,
        };

        let end_lhs = lhs.doc_id_groups.len() / N * N;
        let end_rhs = rhs.doc_id_groups.len() / N * N;
        let a = unsafe { lhs.doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs.doc_id_groups.get_unchecked(..end_rhs) };
        let a_values = if FIRST {
            unsafe { lhs.values.get_unchecked(..end_lhs) }
        } else {
            &lhs.values
        };
        let b_values = unsafe { rhs.values.get_unchecked(..end_rhs) };
        let mut need_to_analyze_msb = false;
        unsafe {
            assume(a.len() % N == 0);
            assume(b.len() % N == 0);
            assume(a_values.len() % N == 0);
            assume(b_values.len() % N == 0);
        }

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

                let vb_values: Simd<u16, N> = _mm_maskz_compress_epi16(
                    mask_b,
                    _mm_load_si128(b_values.as_ptr().add(*rhs_i) as *const _),
                )
                .into();

                if FIRST {
                    let va_values: Simd<u16, N> = _mm_maskz_compress_epi16(
                        mask_a,
                        _mm_load_si128(a_values.as_ptr().add(*lhs_i) as *const _),
                    )
                    .into();

                    _mm_storeu_epi16(
                        values_result.as_mut_ptr().add(*i) as *mut _,
                        ((va_values << 1) & vb_values).into(),
                    );
                } else {
                    _mm_storeu_epi16(
                        values_result.as_mut_ptr().add(*i) as *mut _,
                        (Simd::splat(1) & vb_values).into(),
                    );
                }

                *i += mask_a.count_ones() as usize;
            }

            let last_a = unsafe { *a.get_unchecked(*lhs_i + N - 1) };
            let last_b = unsafe { *b.get_unchecked(*rhs_i + N - 1) };

            if FIRST {
                if last_a <= last_b {
                    unsafe {
                        analyze_msb(va, a_values, *lhs_i, msb_doc_id_groups_result, j);
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
                analyze_msb(va, a_values, *lhs_i, msb_doc_id_groups_result, j);
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
            j,
        );
    }

    #[cfg(not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi2",
        target_feature = "avx512dq",
    )))]
    fn inner_intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        j: &mut usize,
    ) {
        use std::arch::x86_64::_mm256_load_si256;

        let end_lhs = lhs.doc_id_groups.len() / N * N;
        let end_rhs = rhs.doc_id_groups.len() / N * N;
        let a = unsafe { lhs.doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs.doc_id_groups.get_unchecked(..end_rhs) };
        let a_values = unsafe { lhs.values.get_unchecked(..end_lhs) };
        let b_values = unsafe { rhs.values.get_unchecked(..end_rhs) };
        let mut need_to_analyze_msb = false;
        unsafe {
            assume(a.len() % N == 0);
            assume(b.len() % N == 0);
            assume(a_values.len() % N == 0);
            assume(b_values.len() % N == 0);
        }

        while *lhs_i < a.len() && *rhs_i < b.len() {
            let (va, vb) = unsafe {
                let va = _mm256_load_si256(a.as_ptr().add(*lhs_i) as *const _);
                let vb = _mm256_load_si256(b.as_ptr().add(*rhs_i) as *const _);
                (va, vb)
            };

            let (mask_a, mask_b) = unsafe { vp2intersectq(va, vb) };

            if FIRST {
                let mut a_i = 0;
                let mut b_i = 0;
                let mut a_temp = [0u16; N];
                let mut b_temp = [0u16; N];
                for (j, (a_r, b_r)) in mask_a
                    .as_array()
                    .iter()
                    .zip(mask_b.as_array().iter())
                    .enumerate()
                {
                    unsafe {
                        let doc_id_groups = a.get_unchecked(*lhs_i + j);
                        doc_id_groups_result
                            .get_unchecked_mut(*i + a_i)
                            .write(*doc_id_groups);

                        let a_values = *a_values.get_unchecked(*lhs_i + j);
                        *a_temp.get_unchecked_mut(a_i) = a_values;

                        let b_values = *b_values.get_unchecked(*rhs_i + j);
                        *b_temp.get_unchecked_mut(b_i) = b_values;
                    }

                    a_i += (*a_r == u64::MAX) as usize;
                    b_i += (*b_r == u64::MAX) as usize;
                }

                for (j, (a_p, b_p)) in a_temp.iter().zip(b_temp.iter()).enumerate() {
                    unsafe {
                        values_result
                            .get_unchecked_mut(*i + j)
                            .write((a_p << 1) & b_p);
                    }
                }
                *i += a_i;
            } else {
                for (j, b_r) in mask_b.as_array().iter().enumerate() {
                    unsafe {
                        let doc_id_groups = b.get_unchecked(*rhs_i + j);
                        doc_id_groups_result
                            .get_unchecked_mut(*i)
                            .write(*doc_id_groups);

                        let b_values = b_values.get_unchecked(*rhs_i + j);
                        values_result
                            .get_unchecked_mut(*i)
                            .write(*b_values & 1);
                    }
                    *i += (*b_r == u64::MAX) as usize;
                }
            }

            let last_a = unsafe { *a.get_unchecked(*lhs_i + N - 1) };
            let last_b = unsafe { *b.get_unchecked(*rhs_i + N - 1) };

            if FIRST {
                if last_a <= last_b {
                    unsafe {
                        analyze_msb(a, a_values, *lhs_i, msb_doc_id_groups_result, j);
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
            unsafe { analyze_msb(a, a_values, *lhs_i, msb_doc_id_groups_result, j) };
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
            j,
        );
    }

    fn intersection_buffer_size(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> usize {
        lhs.doc_id_groups.len().min(rhs.doc_id_groups.len()) + 1 + N
    }
}
