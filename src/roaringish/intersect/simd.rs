#[allow(unused_imports)]
use std::arch::x86_64::{__m256i, __m512i};
use std::simd::{LaneCount, SupportedLaneCount};
#[allow(unused_imports)]
use std::{
    intrinsics::assume,
    mem::MaybeUninit,
    simd::{cmp::SimdPartialOrd, Simd},
};

use crate::roaringish::{
    BorrowRoaringishPacked, Packed, ADD_ONE_GROUP, MASK_DOC_ID_GROUP, MASK_VALUES,
};

use super::{naive::NaiveIntersect, private::IntersectSeal, Intersect};

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
unsafe fn analyze_msb<L: Packed>(
    va_doc_id_group: Simd<u64, N>,
    va_values: Simd<u64, N>,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) where
    LaneCount<N>: SupportedLaneCount,
    Simd<u64, N>: Into<__m512i>,
{
    use std::arch::x86_64::_mm512_mask_compressstoreu_epi64;

    let vmsb_set = (va_values & Simd::splat(0x8000)).simd_gt(Simd::splat(0));

    let mask = vmsb_set.to_bitmask() as u8;
    let doc_id_groups_plus_one = va_doc_id_group + Simd::splat(ADD_ONE_GROUP);
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
unsafe fn analyze_msb<L: Packed>(
    a: &[L::Packed],
    lhs_i: usize,
    msb_doc_id_groups_result: &mut [MaybeUninit<u64>],
    j: &mut usize,
) {
    let packs = unsafe { a.get_unchecked(lhs_i..(lhs_i + N)) };

    for packed in packs.iter() {
        let doc_id_group: u64 = (*packed).into() & MASK_DOC_ID_GROUP;
        let values: u64 = (*packed).into() & MASK_VALUES;
        unsafe {
            msb_doc_id_groups_result
                .get_unchecked_mut(*j)
                .write(doc_id_group + ADD_ONE_GROUP)
        };
        *j += ((values & 0x8000) > 0) as usize;
    }
}

pub struct SimdIntersect;
impl IntersectSeal for SimdIntersect {}

impl Intersect for SimdIntersect {
    fn intersection_buffer_size<L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,
    ) -> usize {
        // TODO: Is this `+ N` actually needed ? Let's keep it just to be sure
        lhs.packed.len().min(rhs.packed.len()) + 1 + N
    }

    #[inline(always)]
    fn inner_intersect<const FIRST: bool, L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>]>,
        i_packed_result: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>]>,
        i_msb_doc_id_groups_result: &mut usize,
    ) {
        let end_lhs = lhs.packed.len() / N * N;
        let end_rhs = rhs.packed.len() / N * N;
        let a = unsafe { lhs.packed.get_unchecked(..end_lhs) };
        let b = unsafe { rhs.packed.get_unchecked(..end_rhs) };
        let mut need_to_analyze_msb = false;
        unsafe {
            assume(a.len() % N == 0);
            assume(b.len() % N == 0);
        }

        let mask_doc_id_group = Simd::<_, N>::splat(MASK_DOC_ID_GROUP);
        let mask_values = Simd::<_, N>::splat(MASK_VALUES);
        while *lhs_i < a.len() && *rhs_i < b.len() {
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
                let va = _mm512_loadu_epi64(a.as_ptr().add(*lhs_i) as *const _);
                let vb = _mm512_loadu_epi64(b.as_ptr().add(*rhs_i) as *const _);
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
                let va = _mm256_loadu_si256(a.as_ptr().add(*lhs_i) as *const _);
                let vb = _mm256_loadu_si256(b.as_ptr().add(*rhs_i) as *const _);
                (va, vb)
            };
            // -----------------------------------------------

            let va_packed = Simd::from(va);
            let vb_packed = Simd::from(vb);

            let va_doc_id_group = va_packed & mask_doc_id_group;
            let vb_doc_id_group = vb_packed & mask_doc_id_group;
            let va_values = va_packed & mask_values;
            let vb_values = vb_packed & mask_values;

            let (mask_a, mask_b) =
                unsafe { vp2intersectq(va_doc_id_group.into(), vb_doc_id_group.into()) };

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
                    __m128i, _mm512_mask_compressstoreu_epi64, _mm512_maskz_compress_epi64,
                    _mm512_storeu_epi64, _mm_loadu_epi16, _mm_mask_compressstoreu_epi16,
                    _mm_maskz_compress_epi16, _mm_storeu_epi16, _mm_storeu_epi64,
                };
                let doc_id_group: Simd<u64, N> =
                    _mm512_maskz_compress_epi64(mask_a, va_doc_id_group.into()).into();

                let vb_values: Simd<u64, N> =
                    _mm512_maskz_compress_epi64(mask_b, vb_values.into()).into();

                if FIRST {
                    let va_values: Simd<u64, N> =
                        _mm512_maskz_compress_epi64(mask_a, va_values.into()).into();

                    _mm512_storeu_epi64(
                        packed_result.as_mut_ptr().add(*i_packed_result) as *mut _,
                        (doc_id_group | ((va_values << 1) & vb_values)).into(),
                    );
                } else {
                    _mm512_storeu_epi64(
                        packed_result.as_mut_ptr().add(*i_packed_result) as *mut _,
                        (doc_id_group | (Simd::splat(1) & vb_values)).into(),
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
                // TODO: IDK if this is optimal
                if FIRST {
                    let mut a_i = 0;
                    let mut b_i = 0;
                    let mut doc_id_group_temp = [0u64; N];
                    let mut a_values_temp = [0u64; N];
                    let mut b_values_temp = [0u64; N];
                    for ((((a_r, b_r), a_doc_id_group), a_values), b_values) in mask_a
                        .as_array()
                        .iter()
                        .zip(mask_b.as_array())
                        .zip(va_doc_id_group.as_array())
                        .zip(va_values.as_array())
                        .zip(vb_values.as_array())
                    {
                        unsafe {
                            *doc_id_group_temp.get_unchecked_mut(a_i) = *a_doc_id_group;
                            *a_values_temp.get_unchecked_mut(a_i) = *a_values;
                            *b_values_temp.get_unchecked_mut(b_i) = *b_values;
                        }

                        a_i += (*a_r == u64::MAX) as usize;
                        b_i += (*b_r == u64::MAX) as usize;
                    }

                    for (j, ((doc_id_group, a_values), b_values)) in doc_id_group_temp
                        .iter()
                        .zip(a_values_temp)
                        .zip(b_values_temp)
                        .enumerate()
                    {
                        unsafe {
                            packed_result
                                .get_unchecked_mut(*i_packed_result + j)
                                .write(doc_id_group | ((a_values << 1) & b_values));
                        }
                    }

                    *i_packed_result += a_i;
                } else {
                    for ((b_r, b_doc_id_group), b_values) in mask_b
                        .as_array()
                        .iter()
                        .zip(vb_doc_id_group.as_array())
                        .zip(vb_values.as_array())
                    {
                        unsafe {
                            packed_result
                                .get_unchecked_mut(*i_packed_result)
                                .write(*b_doc_id_group | (*b_values & 1));
                        }
                        *i_packed_result += (*b_r == u64::MAX) as usize;
                    }
                }
            }
            // -----------------------------------------------

            let last_a: u64 =
                unsafe { (*a.get_unchecked(*lhs_i + N - 1)).into() & MASK_DOC_ID_GROUP };
            let last_b: u64 =
                unsafe { (*b.get_unchecked(*rhs_i + N - 1)).into() & MASK_DOC_ID_GROUP };

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
                        analyze_msb::<L>(
                            va_doc_id_group,
                            va_values,
                            msb_doc_id_groups_result,
                            i_msb_doc_id_groups_result,
                        );

                        #[cfg(not(all(
                            target_feature = "avx512f",
                            target_feature = "avx512bw",
                            target_feature = "avx512vl",
                            target_feature = "avx512vbmi2",
                            target_feature = "avx512dq",
                        )))]
                        analyze_msb::<L>(
                            a,
                            *lhs_i,
                            msb_doc_id_groups_result,
                            i_msb_doc_id_groups_result,
                        );
                    }
                    *lhs_i += N;
                }
            } else {
                *lhs_i += N * (last_a <= last_b) as usize;
            }
            *rhs_i += N * (last_b <= last_a) as usize;
            need_to_analyze_msb = last_b < last_a;
        }

        if FIRST && need_to_analyze_msb && !(*lhs_i < lhs.packed.len() && *rhs_i < rhs.packed.len())
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
                let va_packed = Simd::from(_mm512_loadu_epi64(a.as_ptr().add(lhs_i) as *const _));
                let va_doc_id_group = va_packed & mask_doc_id_group;
                let va_values = va_packed & mask_values;
                analyze_msb::<L>(
                    va_doc_id_group,
                    va_values,
                    msb_doc_id_groups_result,
                    i_msb_doc_id_groups_result,
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
                analyze_msb::<L>(
                    a,
                    *lhs_i,
                    msb_doc_id_groups_result,
                    i_msb_doc_id_groups_result,
                );
            };
        }

        NaiveIntersect::inner_intersect::<FIRST, L, R>(
            lhs,
            rhs,
            lhs_i,
            rhs_i,
            packed_result,
            i_packed_result,
            msb_doc_id_groups_result,
            i_msb_doc_id_groups_result,
        );
    }
}
