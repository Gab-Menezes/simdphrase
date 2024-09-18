#[allow(unused_imports)]
use std::arch::x86_64::{__m256i, __m512i};
#[allow(unused_imports)]
use std::{
    intrinsics::assume,
    mem::MaybeUninit,
    simd::{cmp::SimdPartialOrd, Simd},
};

use crate::roaringish::{BorrowRoaringishPacked, Packed};

use super::{private::IntersectSeal, Intersect};

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
    return (mask0, mask1);
}

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
)))]
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

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2",
    target_feature = "avx512dq",
))]
#[inline(always)]
unsafe fn avx512_analyze_msb<L: Packed>(
    va: __m512i,
    a_positions: &[L::Position],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) {
    use std::arch::x86_64::{_mm512_mask_compressstoreu_epi64, _mm_loadu_epi16};

    let vpositions: Simd<u16, 8> =
        unsafe { _mm_loadu_epi16(a_positions.as_ptr().add(lhs_i) as *const _).into() };
    let vmsb_set = (vpositions & Simd::splat(0x8000)).simd_gt(Simd::splat(0));

    let mask = vmsb_set.to_bitmask() as u8;
    let va: Simd<u64, 8> = va.into();
    let doc_id_groups_plus_one: Simd<u64, 8> = Simd::splat(1) + va;
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
unsafe fn simple_analyze_msb<L: Packed>(
    a: &[L::DocIdGroup],
    a_positions: &[L::Position],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) {
    let positions = unsafe { a_positions.get_unchecked(lhs_i..(lhs_i + 4)) };

    for (k, pos) in positions.into_iter().enumerate() {
        let doc_id_groups = unsafe { (*a.get_unchecked(lhs_i + k)).into() };
        unsafe {
            msb_doc_id_groups_result
                .get_unchecked_mut(*j)
                .write(doc_id_groups + 1)
        };
        let set = ((*pos).into() & 0x8000) > 0;
        *j += set as usize;
    }
}

pub struct SimdIntersect;
impl IntersectSeal for SimdIntersect {}

impl Intersect for SimdIntersect {
    fn intersect<const FIRST: bool, L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,
    ) -> (Vec<u64>, Vec<u16>, Vec<u64>) {
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

        let lhs_positions = lhs.positions;
        let rhs_positions = rhs.positions;
        let lhs_doc_id_groups = lhs.doc_id_groups;
        let rhs_doc_id_groups = rhs.doc_id_groups;

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        // TODO: Is this `+ N` actually needed ? Let's keep it just to be sure
        let min = lhs_doc_id_groups.len().min(rhs_doc_id_groups.len()) + 1 + N;
        let mut i = 0;
        let mut j = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
        let mut positions_intersect: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(min);
        let mut msb_doc_id_groups_result: Box<[MaybeUninit<u64>]> = if FIRST {
            Box::new_uninit_slice(lhs_doc_id_groups.len() + 1)
        } else {
            Box::new_uninit_slice(0)
        };

        let end_lhs = lhs_doc_id_groups.len() / N * N;
        let end_rhs = rhs_doc_id_groups.len() / N * N;
        let a = unsafe { lhs_doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs_doc_id_groups.get_unchecked(..end_rhs) };
        let a_positions = unsafe { lhs_positions.get_unchecked(..end_lhs) };
        let b_positions = unsafe { rhs_positions.get_unchecked(..end_rhs) };
        let mut need_to_analyze_msb = false;
        unsafe {
            assume(a.len() % N == 0);
            assume(b.len() % N == 0);
            assume(a_positions.len() % N == 0);
            assume(b_positions.len() % N == 0);
        }

        while lhs_i < a.len() && rhs_i < b.len() {
            // --------------------- LOAD ---------------------
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi2",
                target_feature = "avx512dq",
            ))]
            let (va, vb) = unsafe {
                use std::arch::x86_64::_mm512_loadu_epi64;
                let va = _mm512_loadu_epi64(a.as_ptr().add(lhs_i) as *const _);
                let vb = _mm512_loadu_epi64(b.as_ptr().add(rhs_i) as *const _);
                (va, vb)
            };

            #[cfg(not(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi2",
                target_feature = "avx512dq",
            )))]
            let (va, vb) = unsafe {
                use std::arch::x86_64::_mm256_loadu_si256;
                let va = _mm256_loadu_si256(a.as_ptr().add(lhs_i) as *const _);
                let vb = _mm256_loadu_si256(b.as_ptr().add(rhs_i) as *const _);
                (va, vb)
            };
            // -----------------------------------------------

            let (mask_a, mask_b) = unsafe { vp2intersectq(va, vb) };

            // --------------------- STORE ---------------------
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi2",
                target_feature = "avx512dq",
            ))]
            unsafe {
                use std::arch::x86_64::{
                    __m128i, _mm512_mask_compressstoreu_epi64, _mm_loadu_epi16,
                    _mm_mask_compressstoreu_epi16, _mm_maskz_compress_epi16, _mm_storeu_epi16,
                };
                // TODO: avoid compressstore on zen4
                _mm512_mask_compressstoreu_epi64(
                    doc_id_groups_result.as_mut_ptr().add(i) as *mut _,
                    mask_a,
                    va,
                );

                let vb_positions: Simd<u16, N> = _mm_maskz_compress_epi16(
                    mask_b,
                    _mm_loadu_epi16(b_positions.as_ptr().add(rhs_i) as *const _),
                )
                .into();

                if FIRST {
                    let va_positions: Simd<u16, N> = _mm_maskz_compress_epi16(
                        mask_a,
                        _mm_loadu_epi16(a_positions.as_ptr().add(lhs_i) as *const _),
                    )
                    .into();

                    _mm_storeu_epi16(
                        positions_intersect.as_mut_ptr().add(i) as *mut _,
                        ((va_positions << 1) & vb_positions).into(),
                    );
                } else {
                    _mm_storeu_epi16(
                        positions_intersect.as_mut_ptr().add(i) as *mut _,
                        (Simd::splat(1) & vb_positions).into(),
                    );
                }

                i += mask_a.count_ones() as usize;
            }

            #[cfg(not(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi2",
                target_feature = "avx512dq",
            )))]
            {
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
                            let doc_id_groups = a.get_unchecked(lhs_i + j);
                            doc_id_groups_result
                                .get_unchecked_mut(i + a_i)
                                .write((*doc_id_groups).into());

                            let a_position = *a_positions.get_unchecked(lhs_i + j);
                            *a_temp.get_unchecked_mut(i + a_i) = a_position.into();

                            let b_positions = *b_positions.get_unchecked(rhs_i + j);
                            *b_temp.get_unchecked_mut(i + b_i) = b_positions.into();
                        }

                        a_i += (*a_r == u64::MAX) as usize;
                        b_i += (*b_r == u64::MAX) as usize;
                    }

                    for (j, (a_p, b_p)) in a_temp.iter().zip(b_temp.iter()).enumerate() {
                        unsafe {
                            positions_intersect
                                .get_unchecked_mut(i + j)
                                .write((a_p << 1) & b_p);
                        }
                    }

                    i += a_i;
                } else {
                    for (j, b_r) in mask_b.as_array().iter().enumerate() {
                        unsafe {
                            let doc_id_groups = a.get_unchecked(lhs_i + j);
                            doc_id_groups_result
                                .get_unchecked_mut(i)
                                .write((*doc_id_groups).into());

                            let b_positions = b_positions.get_unchecked(rhs_i + j);
                            positions_intersect
                                .get_unchecked_mut(i)
                                .write((*b_positions).into() & 1);
                        }
                        i += (*b_r == u64::MAX) as usize;
                    }
                }
            }
            // -----------------------------------------------

            let last_a = unsafe { (*a.get_unchecked(lhs_i + N - 1)).into() };
            let last_b = unsafe { (*b.get_unchecked(rhs_i + N - 1)).into() };

            if FIRST {
                if last_a <= last_b {
                    unsafe {
                        #[cfg(all(
                            target_feature = "avx512f",
                            target_feature = "avx512bw",
                            target_feature = "avx512vl",
                            target_feature = "avx512vbmi2",
                            target_feature = "avx512dq",
                        ))]
                        avx512_analyze_msb::<L>(
                            va,
                            a_positions,
                            lhs_i,
                            &mut msb_doc_id_groups_result,
                            &mut j,
                        );

                        #[cfg(not(all(
                            target_feature = "avx512f",
                            target_feature = "avx512bw",
                            target_feature = "avx512vl",
                            target_feature = "avx512vbmi2",
                            target_feature = "avx512dq",
                        )))]
                        simple_analyze_msb::<L>(
                            a,
                            a_positions,
                            lhs_i,
                            &mut msb_doc_id_groups_result,
                            &mut j,
                        );
                    }
                    lhs_i += N;
                }
            } else {
                lhs_i += N * (last_a <= last_b) as usize;
            }
            rhs_i += N * (last_b <= last_a) as usize;
            need_to_analyze_msb = last_b < last_a;
        }

        if FIRST {
            if need_to_analyze_msb
                && !(lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len())
            {
                #[cfg(all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi2",
                    target_feature = "avx512dq",
                ))]
                unsafe {
                    use std::arch::x86_64::_mm512_loadu_epi64;
                    let va = _mm512_loadu_epi64(a.as_ptr().add(lhs_i) as *const _);
                    avx512_analyze_msb::<L>(
                        va,
                        a_positions,
                        lhs_i,
                        &mut msb_doc_id_groups_result,
                        &mut j,
                    );
                };

                #[cfg(not(all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi2",
                    target_feature = "avx512dq",
                )))]
                unsafe {
                    simple_analyze_msb::<L>(
                        a,
                        a_positions,
                        lhs_i,
                        &mut msb_doc_id_groups_result,
                        &mut j,
                    )
                };
            }
        }

        // for the remaining elements we can do a 2 pointer approach
        // since the 2 slices will be small
        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { (*lhs_doc_id_groups.get_unchecked(lhs_i)).into() };
            let rhs_doc_id_groups = unsafe { (*rhs_doc_id_groups.get_unchecked(rhs_i)).into() };
            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(lhs_doc_id_groups);
                    let rhs = (*rhs_positions.get_unchecked(rhs_i)).into();
                    if FIRST {
                        let lhs = (*lhs_positions.get_unchecked(lhs_i)).into();
                        positions_intersect
                            .get_unchecked_mut(i)
                            .write((lhs << 1) & rhs);

                        msb_doc_id_groups_result
                            .get_unchecked_mut(j)
                            .write(lhs_doc_id_groups + 1);
                        j += (lhs & 0x8000 > 1) as usize;
                    }
                };

                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                rhs_i += 1;
            } else {
                if FIRST {
                    unsafe {
                        let lhs = (*lhs_positions.get_unchecked(lhs_i)).into();
                        msb_doc_id_groups_result
                            .get_unchecked_mut(j)
                            .write(lhs_doc_id_groups + 1);
                        j += (lhs & 0x8000 > 1) as usize;
                    }
                }

                lhs_i += 1;
            }
        }

        let doc_id_groups_result_ptr = Box::into_raw(doc_id_groups_result) as *mut _;
        let positions_intersect_ptr = Box::into_raw(positions_intersect) as *mut _;
        let msb_doc_id_groups_result_ptr = Box::into_raw(msb_doc_id_groups_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(doc_id_groups_result_ptr, i, min),
                Vec::from_raw_parts(positions_intersect_ptr, i, min),
                if FIRST {
                    Vec::from_raw_parts(
                        msb_doc_id_groups_result_ptr,
                        j,
                        lhs_doc_id_groups.len() + 1,
                    )
                } else {
                    Vec::from_raw_parts(msb_doc_id_groups_result_ptr, 0, 0)
                },
            )
        }
    }
}
