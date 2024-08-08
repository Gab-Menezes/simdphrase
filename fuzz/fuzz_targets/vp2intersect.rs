#![feature(stdarch_x86_avx512)]
#![feature(portable_simd)]

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::arch::x86_64::{
    __m512i, 
    _mm512_loadu_epi64, 
    _mm512_cmpeq_epi64_mask, 
    _mm512_shuffle_epi32, 
    _mm512_alignr_epi32, 
    _MM_PERM_BADC, 
    __mmask8,
    _mm256_loadu_si256,
    __m256i,
    _mm256_shuffle_epi32,
    _mm256_or_si256,
    _mm256_permute2x128_si256,
    _mm256_cmpeq_epi64
};
use std::arch::asm;
use std::simd::Mask;

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vp2intersect"
))]
unsafe fn native_vp2intersectq_avx512(a: __m512i, b: __m512i) -> (u8, u8) {
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
    target_feature = "avx512dq",
    target_feature = "avx512bw",
))]
unsafe fn emulate_vp2intersectq_avx512(a: __m512i, b: __m512i) -> (u8, u8) {
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

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vp2intersect"
))]
unsafe fn native_vp2intersectq_avx2(a: __m256i, b: __m256i) -> (u8, u8) {
    let mut mask0: __mmask8;
    let mut mask1: __mmask8;
    asm!(
        "vp2intersectq k2, {0}, {1}",
        in(ymm_reg) a,
        in(ymm_reg) b,
        out("k2") mask0,
        out("k3") mask1,
        options(pure, nomem, nostack),
    );

    (mask0, mask1)
}

unsafe fn emulate_vp2intersectq_avx2(a: __m256i, b: __m256i) -> (u8, u8) {
    let a1 = _mm256_permute2x128_si256(a, a, 1);
    let b1 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);

    let m00 = _mm256_cmpeq_epi64(a, b);
    let m01 = _mm256_cmpeq_epi64(a, b1);
    let m10 = _mm256_cmpeq_epi64(a1, b);
    let m11 = _mm256_cmpeq_epi64(a1, b1);

    let l = _mm256_or_si256(m00, m01);
    let h = _mm256_or_si256(m10, m11);
    let mask0 = _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
    let mask0 = unsafe { std::mem::transmute::<__m256i, Mask<i64, 4>>(mask0).to_bitmask() as u8 };

    let l = _mm256_or_si256(m00, m10);
    let h = _mm256_or_si256(m01, m11);
    let mask1 = _mm256_or_si256(l, _mm256_shuffle_epi32(h, _MM_PERM_BADC));
    let mask1 = unsafe { std::mem::transmute::<__m256i, Mask<i64, 4>>(mask1).to_bitmask() as u8 };
    (mask0, mask1)
}

fuzz_target!(|v: ([i64; 8], [i64; 8])| {
    unsafe {
        let a = _mm512_loadu_epi64(&v.0 as *const _);
        let b = _mm512_loadu_epi64(&v.1 as *const _);
        assert_eq!(emulate_vp2intersectq_avx512(a, b), native_vp2intersectq_avx512(a, b));

        let a = _mm256_loadu_si256(&v.0 as *const _ as *const _);
        let b = _mm256_loadu_si256(&v.1 as *const _ as *const _);
        assert_eq!(emulate_vp2intersectq_avx2(a, b), native_vp2intersectq_avx2(a, b));
    }
});
