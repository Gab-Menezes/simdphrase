use std::mem::MaybeUninit;

use crate::{Stats, allocator::Aligned64};

use super::{ADD_ONE_GROUP, Aligned, BorrowRoaringishPacked};

pub mod gallop_first;
pub mod gallop_second;
pub mod naive;
pub mod simd;

mod private {
    pub trait IntersectSeal {}
}

/// Allows a type to be used as an intersection algorithm when searching.
pub trait Intersection: Intersect {}

/// Necessary functions for an intersection algorithm.
///
/// The intersection is done in two phases that's why
/// the function have a `FIRST` const generic.
pub trait Intersect: private::IntersectSeal {
    /// Responsible for allocating the result buffers
    /// and compute the necessary values before starting
    /// the intersection.
    #[inline(never)]
    fn intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
        lhs_len: u32,

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

        let add_to_group = (lhs_len / 16) as u64 * ADD_ONE_GROUP;
        let lhs_len = (lhs_len % 16) as u16;

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
            add_to_group,
            lhs_len,
            msb_mask,
            lsb_mask,
            stats,
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

    /// Performs the intersection.
    ///
    /// `msb_packed_result` has 0 capacity if `FIRST` is false.
    #[allow(clippy::too_many_arguments)]
    fn inner_intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        i: &mut usize,

        msb_packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        j: &mut usize,

        add_to_group: u64,
        lhs_len: u16,
        msb_mask: u16,
        lsb_mask: u16,

        stats: &Stats,
    );

    /// Size of the buffer needed to store the intersection.
    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize;
}
