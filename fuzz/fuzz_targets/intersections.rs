#![no_main]

use libfuzzer_sys::fuzz_target;
use phrase_search::RoaringishPacked;
use phrase_search::BorrowRoaringishPacked;
use phrase_search::naive::NaiveIntersect;
use phrase_search::naive::UnrolledNaiveIntersect;
use phrase_search::Intersect;
use phrase_search::gallop::GallopIntersect;
use phrase_search::simd::SimdIntersect;

fn compare(lhs: &(Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>), rhs: &(Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>)) {
    assert_eq!(lhs.0, rhs.0);
    assert_eq!(lhs.1, rhs.1);
    assert_eq!(lhs.2, rhs.2);
    assert!(lhs.3.len() <= rhs.3.len());
    assert_eq!(lhs.3, rhs.3[..lhs.3.len()]);
    for v in rhs.3.windows(2) {
        assert!(v[0] < v[1]);
    }
}

fuzz_target!(|r: (RoaringishPacked, RoaringishPacked)| {
    let lhs = BorrowRoaringishPacked::new(&r.0);
    let rhs = BorrowRoaringishPacked::new(&r.1);
    
    let naive = NaiveIntersect::intersect::<true>(&lhs, &rhs);
    // let unrolled_naive = UnrolledNaiveIntersect::intersect::<true>(&lhs, &rhs);
    // let gallop = GallopIntersect::intersect::<true>(&lhs, &rhs);
    // let simd_cmov = SimdIntersectCMOV::intersect::<true>(&lhs, &rhs);
    let simd = SimdIntersect::intersect::<true>(&lhs, &rhs);

    // assert_eq!(naive, gallop);
    // assert_eq!(naive, unrolled_naive);
    // assert_eq!(naive, simd_cmov);
    compare(&naive, &simd);
});
