#![feature(allocator_api)]

#![no_main]

use libfuzzer_sys::fuzz_target;
use phrase_search::AlignedRoaringishPacked;
use phrase_search::AlignedBorrowRoaringishPacked;
use phrase_search::naive::NaiveIntersect;
use phrase_search::Intersect;
use phrase_search::simd::SimdIntersect;
use phrase_search::Aligned64;

fn compare(lhs: &(Vec<u64, Aligned64>, Vec<u16, Aligned64>, Vec<u64, Aligned64>), rhs: &(Vec<u64, Aligned64>, Vec<u16, Aligned64>, Vec<u64, Aligned64>)) {
    assert_eq!(lhs.0, rhs.0);
    assert_eq!(lhs.1, rhs.1);
    assert!(lhs.2.len() <= rhs.2.len());
    assert_eq!(lhs.2, rhs.2[..lhs.2.len()]);
    for v in rhs.2.windows(2) {
        assert!(v[0] < v[1]);
    }
}

fuzz_target!(|r: (AlignedRoaringishPacked, AlignedRoaringishPacked)| {
    let lhs = AlignedBorrowRoaringishPacked::new(&r.0);
    let rhs = AlignedBorrowRoaringishPacked::new(&r.1);
    
    let naive = NaiveIntersect::intersect::<true>(&lhs, &rhs);
    let simd = SimdIntersect::intersect::<true>(&lhs, &rhs);

    compare(&naive, &simd);

    let naive = NaiveIntersect::intersect::<false>(&lhs, &rhs);
    let simd = SimdIntersect::intersect::<false>(&lhs, &rhs);

    compare(&naive, &simd);
});
