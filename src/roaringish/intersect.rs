// use super::private::BorrowRoaringishPacked;

use std::mem::MaybeUninit;

use crate::{allocator::Aligned64, Stats};

use super::{Aligned, BorrowRoaringishPacked};

pub mod naive;
pub mod simd;

mod private {
    pub trait IntersectSeal {}
}

pub trait Intersect: private::IntersectSeal {
    #[inline(always)]
    fn intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
        lhs_len: u16,

        stats: &Stats,
    ) -> (Vec<u64, Aligned64>, Vec<u64, Aligned64>) {
        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let buffer_size = Self::intersection_buffer_size(lhs, rhs);

        let mut i = 0;
        let mut packed_result: Box<[MaybeUninit<u64>], Aligned64> =
            Box::new_uninit_slice_in(buffer_size, Aligned64::default());

        let mut j = 0;
        let mut msb_packed_result: Box<[MaybeUninit<u64>], Aligned64> = if FIRST {
            Box::new_uninit_slice_in(lhs.0.len() + 1, Aligned64::default())
        } else {
            Box::new_uninit_slice_in(0, Aligned64::default())
        };

        let msb_mask = !(u16::MAX >> lhs_len);
        let lsb_mask = !(u16::MAX << lhs_len);

        Self::inner_intersect::<FIRST>(
            lhs,
            rhs,
            &mut lhs_i,
            &mut rhs_i,
            &mut packed_result,
            &mut i,
            &mut msb_packed_result,
            &mut j,
            lhs_len,
            msb_mask,
            lsb_mask,

            stats
        );

        let (packed_result_ptr, a0) = Box::into_raw_with_allocator(packed_result);
        let (msb_packed_result_ptr, a1) = Box::into_raw_with_allocator(msb_packed_result);
        unsafe {
            (
                Vec::from_raw_parts_in(packed_result_ptr as *mut _, i, buffer_size, a0),
                if FIRST {
                    Vec::from_raw_parts_in(msb_packed_result_ptr as *mut _, j, lhs.0.len() + 1, a1)
                } else {
                    Vec::from_raw_parts_in(msb_packed_result_ptr as *mut _, 0, 0, a1)
                },
            )
        }
    }

    fn inner_intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        i: &mut usize,

        msb_packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        j: &mut usize,

        lhs_len: u16,
        msb_mask: u16,
        lsb_mask: u16,

        stats: &Stats,
    );

    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize;
}
