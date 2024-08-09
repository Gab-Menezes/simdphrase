#[allow(unused_imports)]
use std::arch::x86_64::{__m256i, __m512i};
use std::{
    intrinsics::assume,
    mem::MaybeUninit,
    simd::{cmp::SimdPartialOrd, Simd},
};

use crate::roaringish::BorrowRoaringishPacked;

use super::{private::IntersectSeal, Intersect};

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vp2intersect"
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

#[cfg(all(target_feature = "avx512f", not(target_feature = "avx512vp2intersect")))]
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
    return (mask0, mask1);
}

#[cfg(all(not(target_feature = "avx512f")))]
#[inline(always)]
unsafe fn vp2intersectq(a: __m256i, b: __m256i) -> (Simd<u64, 4>, Simd<u64, 4>) {
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

pub struct Vp2Intersectq;
impl IntersectSeal for Vp2Intersectq {}

impl Intersect for Vp2Intersectq {
    fn intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> (Vec<u64>, Vec<u16>, Vec<u16>) {
        let lhs_positions = lhs.positions;
        let rhs_positions = rhs.positions;
        let lhs_doc_id_groups = lhs.doc_id_groups;
        let rhs_doc_id_groups = rhs.doc_id_groups;

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let min = lhs_doc_id_groups.len().min(rhs_doc_id_groups.len()) + 1;
        let mut i = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
        let mut lhs_positions_result: Box<[MaybeUninit<u16>]> = if FIRST {
            Box::new_uninit_slice(min)
        } else {
            Box::new_uninit_slice(0)
        };
        let mut rhs_positions_result: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(min);

        #[cfg(target_feature = "avx512f")]
        const N: usize = 8;
        #[cfg(not(target_feature = "avx512f"))]
        const N: usize = 4;

        let end_lhs = lhs_doc_id_groups.len() / N * N;
        let end_rhs = rhs_doc_id_groups.len() / N * N;
        let a = unsafe { lhs_doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs_doc_id_groups.get_unchecked(..end_rhs) };
        let a_positions = unsafe { lhs_positions.get_unchecked(..end_lhs) };
        let b_positions = unsafe { rhs_positions.get_unchecked(..end_rhs) };
        unsafe {
            assume(a.len() % N == 0);
            assume(b.len() % N == 0);
            assume(a_positions.len() % N == 0);
            assume(b_positions.len() % N == 0);
        }

        while lhs_i < a.len() && rhs_i < b.len() {
            #[cfg(target_feature = "avx512f")]
            let (va, vb) = unsafe {
                use std::arch::x86_64::_mm512_loadu_epi64;
                let va = _mm512_loadu_epi64(a.as_ptr().add(lhs_i) as *const _);
                let vb = _mm512_loadu_epi64(b.as_ptr().add(rhs_i) as *const _);
                (va, vb)
            };

            #[cfg(not(target_feature = "avx512f"))]
            let (va, vb) = unsafe {
                use std::arch::x86_64::_mm256_loadu_si256;
                let va = _mm256_loadu_si256(a.as_ptr().add(lhs_i) as *const _);
                let vb = _mm256_loadu_si256(b.as_ptr().add(rhs_i) as *const _);
                (va, vb)
            };

            let (mask_a, mask_b) = unsafe { vp2intersectq(va, vb) };

            #[cfg(target_feature = "avx512f")]
            unsafe {
                use std::arch::x86_64::{
                    _mm512_loadu_epi16, _mm512_mask_compressstoreu_epi16,
                    _mm512_mask_compressstoreu_epi64,
                };
                _mm512_mask_compressstoreu_epi64(
                    doc_id_groups_result.as_mut_ptr().add(i) as *mut _,
                    mask_a,
                    va,
                );

                let vb_positions = _mm512_loadu_epi16(b_positions.as_ptr().add(rhs_i) as *const _);
                _mm512_mask_compressstoreu_epi16(
                    rhs_positions_result.as_mut_ptr().add(i) as *mut _,
                    mask_b as u32,
                    vb_positions,
                );

                if FIRST {
                    let va_positions =
                        _mm512_loadu_epi16(a_positions.as_ptr().add(lhs_i) as *const _);
                    _mm512_mask_compressstoreu_epi16(
                        lhs_positions_result.as_mut_ptr().add(i) as *mut _,
                        mask_a as u32,
                        va_positions,
                    );
                }

                i += mask_a.count_ones() as usize;
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                let mut c = 0;
                for (j, r) in mask_a.as_array().iter().enumerate() {
                    let doc_id_groups = unsafe { a.get_unchecked(lhs_i + j) };
                    unsafe {
                        doc_id_groups_result
                            .get_unchecked_mut(i + c)
                            .write(*doc_id_groups)
                    };
                    if FIRST {
                        let positions = unsafe { a_positions.get_unchecked(lhs_i + j) };
                        unsafe {
                            lhs_positions_result
                                .get_unchecked_mut(i + c)
                                .write(*positions)
                        };
                    }
                    c += (*r == u64::MAX) as usize;
                }

                for (j, r) in mask_b.as_array().iter().enumerate() {
                    let positions = unsafe { b_positions.get_unchecked(rhs_i + j) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(*positions) };
                    i += (*r == u64::MAX) as usize;
                }
            }

            let last_a = unsafe { *a.as_ptr().add(lhs_i + N - 1) };
            let last_b = unsafe { *b.as_ptr().add(rhs_i + N - 1) };

            lhs_i += N * (last_a <= last_b) as usize;
            rhs_i += N * (last_b <= last_a) as usize;

            // let va: Simd<u64, N> = va.into();
            // let vb: Simd<u64, N> = vb.into();

            // let last_va = Simd::splat(last_a);
            // let last_vb = Simd::splat(last_b);

            // lhs_i += 64 - va.simd_le(last_vb).to_bitmask().leading_zeros() as usize;
            // rhs_i += 64 - vb.simd_le(last_va).to_bitmask().leading_zeros() as usize;
        }

        // for the remaining elements we can do a 2 pointer approach
        // since the 2 slices will be small
        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { lhs_doc_id_groups.get_unchecked(lhs_i) };
            let rhs_doc_id_groups = unsafe { rhs_doc_id_groups.get_unchecked(rhs_i) };
            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(*lhs_doc_id_groups)
                };
                if FIRST {
                    let lhs = unsafe { *lhs_positions.get_unchecked(lhs_i) };
                    unsafe { lhs_positions_result.get_unchecked_mut(i).write(lhs) };

                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                } else {
                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                }

                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                rhs_i += 1;
            } else {
                lhs_i += 1;
            }
        }

        let doc_id_groups_result_ptr = Box::into_raw(doc_id_groups_result) as *mut _;
        let lhs_positions_result_ptr = Box::into_raw(lhs_positions_result) as *mut _;
        let rhs_positions_result_ptr = Box::into_raw(rhs_positions_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(doc_id_groups_result_ptr, i, min),
                if FIRST {
                    Vec::from_raw_parts(lhs_positions_result_ptr, i, min)
                } else {
                    Vec::from_raw_parts(lhs_positions_result_ptr, 0, 0)
                },
                Vec::from_raw_parts(rhs_positions_result_ptr, i, min),
            )
        }
    }
}
