#![no_main]

use libfuzzer_sys::fuzz_target;
use phrase_search::RoaringishPacked;
use phrase_search::BorrowRoaringishPacked;
use phrase_search::naive::NaiveIntersect;
use phrase_search::Intersect;
use phrase_search::simd::SimdIntersect;

fn compare(lhs: &(Vec<u64>, Vec<u16>, Vec<u64>), rhs: &(Vec<u64>, Vec<u16>, Vec<u64>)) {
    assert_eq!(lhs.0, rhs.0);
    assert_eq!(lhs.1, rhs.1);
    assert_eq!(lhs.2, rhs.2);
    assert!(lhs.2.len() <= rhs.2.len());
    assert_eq!(lhs.2, rhs.2[..lhs.2.len()]);
    for v in rhs.2.windows(2) {
        assert!(v[0] < v[1]);
    }
}

fuzz_target!(|r: (RoaringishPacked, RoaringishPacked)| {
    let lhs = BorrowRoaringishPacked::new(&r.0);
    let rhs = BorrowRoaringishPacked::new(&r.1);
    
    let naive = NaiveIntersect::intersect::<true, _, _>(&lhs, &rhs);
    let simd = SimdIntersect::intersect::<true, _, _>(&lhs, &rhs);

    compare(&naive, &simd);

    let naive = NaiveIntersect::intersect::<false, _, _>(&lhs, &rhs);
    let simd = SimdIntersect::intersect::<false, _, _>(&lhs, &rhs);

    compare(&naive, &simd);
});
