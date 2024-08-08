#![feature(stdarch_x86_avx512)]

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::arch::x86_64::{__m512i, _mm512_loadu_epi64, _mm512_cmpeq_epi64_mask, _mm512_shuffle_epi32, _mm512_alignr_epi32, _MM_PERM_BADC, __mmask8};
use std::arch::asm;

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vp2intersect"
))]
unsafe fn native_vp2intersectq(a: __m512i, b: __m512i) -> (__mmask8, __mmask8) {
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
unsafe fn emulate_vp2intersectq(a: __m512i, b: __m512i) -> (__mmask8, __mmask8) {
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

fuzz_target!(|v: ([i64; 8], [i64; 8])| {
    unsafe {
        let a = _mm512_loadu_epi64(&v.0 as *const _);
        let b = _mm512_loadu_epi64(&v.1 as *const _);
        assert_eq!(emulate_vp2intersectq(a, b), native_vp2intersectq(a, b));
    }
});
