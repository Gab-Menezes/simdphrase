// use super::private::BorrowRoaringishPacked;

use super::{BorrowRoaringishPacked, Packed};

pub mod naive;
pub mod simd;

mod private {
    pub trait IntersectSeal {}
}

pub trait Intersect: private::IntersectSeal {
    fn intersect<const FIRST: bool, L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,
    ) -> (Vec<u64>, Vec<u16>, Vec<u64>);
}
