// use super::private::BorrowRoaringishPacked;

use std::mem::MaybeUninit;

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
    ) -> (Vec<u64>, Vec<u16>, Vec<u64>) {
        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let buffer_size = Self::intersection_buffer_size(lhs, rhs);

        let mut i = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(buffer_size);
        let mut values_result: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(buffer_size);

        let mut j = 0;
        let mut msb_doc_id_groups_result: Box<[MaybeUninit<u64>]> = if FIRST {
            Box::new_uninit_slice(lhs.doc_id_groups.len() + 1)
        } else {
            Box::new_uninit_slice(0)
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

        let doc_id_groups_result_ptr = Box::into_raw(doc_id_groups_result) as *mut _;
        let values_result_ptr = Box::into_raw(values_result) as *mut _;
        let msb_doc_id_groups_result_ptr = Box::into_raw(msb_doc_id_groups_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(doc_id_groups_result_ptr, i, buffer_size),
                Vec::from_raw_parts(values_result_ptr, i, buffer_size),
                if FIRST {
                    Vec::from_raw_parts(
                        msb_doc_id_groups_result_ptr,
                        j,
                        lhs.doc_id_groups.len() + 1,
                    )
                } else {
                    Vec::from_raw_parts(msb_doc_id_groups_result_ptr, 0, 0)
                },
            )
        }
    }

    fn inner_intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>]>,
        values_result: &mut Box<[MaybeUninit<u16>]>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>]>,
        j: &mut usize,
    );

    fn intersection_buffer_size(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> usize;
}
