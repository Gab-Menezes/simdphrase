#![feature(allocator_api)]

#![no_main]

use libfuzzer_sys::fuzz_target;
use phrase_search::RoaringishPacked;
use phrase_search::BorrowRoaringishPacked;
use phrase_search::naive::NaiveIntersect;
use phrase_search::Intersect;
use phrase_search::simd::SimdIntersect;
use phrase_search::Aligned64;

fn compare(lhs: &(Vec<u64, Aligned64>, Vec<u16, Aligned64>, Vec<u64, Aligned64>, Vec<u16, Aligned64>), rhs: &(Vec<u64, Aligned64>, Vec<u16, Aligned64>, Vec<u64, Aligned64>, Vec<u16, Aligned64>)) {
    assert_eq!(lhs.0, rhs.0);
    assert_eq!(lhs.1, rhs.1);

    assert!(lhs.2.len() <= rhs.2.len());
    assert_eq!(lhs.2, rhs.2[..lhs.2.len()]);
    for v in rhs.2.windows(2) {
        assert!(v[0] < v[1]);
    }

    assert!(lhs.3.len() <= rhs.3.len());
    assert_eq!(lhs.3, rhs.3[..lhs.3.len()]);
    for v in &rhs.3 {
        assert!(*v != 0);
    }
}

fuzz_target!(|r: (RoaringishPacked, RoaringishPacked, u16)| {
    let lhs = BorrowRoaringishPacked::new(&r.0);
    let rhs = BorrowRoaringishPacked::new(&r.1);

    let l = (r.2 % 4) + 1;

    // let naive = NaiveIntersect::intersect::<true>(lhs, rhs, l);
    // let simd = SimdIntersect::intersect::<true>(lhs, rhs, l);
    // compare(&naive, &simd);

    let naive = NaiveIntersect::intersect::<false>(lhs, rhs, l);
    let simd = SimdIntersect::intersect::<false>(lhs, rhs, l);
    compare(&naive, &simd);
});
