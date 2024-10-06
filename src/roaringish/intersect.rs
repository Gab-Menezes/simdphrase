// use super::private::BorrowRoaringishPacked;

use std::mem::MaybeUninit;

use crate::allocator::Aligned64;

use super::BorrowRoaringishPacked;

pub mod naive;
pub mod simd;

mod private {
    pub trait IntersectSeal {}
}

pub trait Intersect: private::IntersectSeal {
    fn intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> (
        Vec<u64, Aligned64>,
        Vec<u16, Aligned64>,
        Vec<u64, Aligned64>,
    ) {
        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let buffer_size = Self::intersection_buffer_size(lhs, rhs);

        let mut i = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>], Aligned64> =
            Box::new_uninit_slice_in(buffer_size, Aligned64::default());
        let mut values_result: Box<[MaybeUninit<u16>], Aligned64> =
            Box::new_uninit_slice_in(buffer_size, Aligned64::default());

        let mut j = 0;
        let mut msb_doc_id_groups_result: Box<[MaybeUninit<u64>], Aligned64> = if FIRST {
            Box::new_uninit_slice_in(lhs.doc_id_groups.len() + 1, Aligned64::default())
        } else {
            Box::new_uninit_slice_in(0, Aligned64::default())
        };

        Self::inner_intersect::<FIRST>(
            lhs,
            rhs,
            &mut lhs_i,
            &mut rhs_i,
            &mut doc_id_groups_result,
            &mut values_result,
            &mut i,
            &mut msb_doc_id_groups_result,
            &mut j,
        );

        let (doc_id_groups_result_ptr, a0) = Box::into_raw_with_allocator(doc_id_groups_result);
        let (values_result_ptr, a1) = Box::into_raw_with_allocator(values_result);
        let (msb_doc_id_groups_result_ptr, a2) =
            Box::into_raw_with_allocator(msb_doc_id_groups_result);
        unsafe {
            (
                Vec::from_raw_parts_in(doc_id_groups_result_ptr as *mut _, i, buffer_size, a0),
                Vec::from_raw_parts_in(values_result_ptr as *mut _, i, buffer_size, a1),
                if FIRST {
                    Vec::from_raw_parts_in(
                        msb_doc_id_groups_result_ptr as *mut _,
                        j,
                        lhs.doc_id_groups.len() + 1,
                        a2,
                    )
                } else {
                    Vec::from_raw_parts_in(msb_doc_id_groups_result_ptr as *mut _, 0, 0, a2)
                },
            )
        }
    }

    fn inner_intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        j: &mut usize,
    );

    fn intersection_buffer_size(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> usize;
}
