// use super::private::BorrowRoaringishPacked;

use super::BorrowRoaringishPacked;

pub mod naive;
pub mod gallop;
pub mod simd;
pub mod vp2intersectq;

mod private {
    pub trait IntersectSeal {}
}

pub trait Intersect: private::IntersectSeal {
    fn intersect<const FIRST: bool>(lhs: &BorrowRoaringishPacked, rhs: &BorrowRoaringishPacked) -> (Vec<u64>, Vec<u16>, Vec<u16>);
}