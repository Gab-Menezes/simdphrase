// use super::private::BorrowRoaringishPacked;

use std::mem::MaybeUninit;

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
    ) -> (Vec<u64>, Vec<u64>) {
        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let min = Self::intersection_buffer_size(lhs, rhs);
        let mut i_packed_result = 0;
        let mut i_msb_doc_id_groups_result = 0;
        let mut packed_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
        let mut msb_doc_id_groups_result: Box<[MaybeUninit<u64>]> = if FIRST {
            Box::new_uninit_slice(lhs.packed.len() + 1)
        } else {
            Box::new_uninit_slice(0)
        };

        Self::inner_intersect::<FIRST, L, R>(
            lhs,
            rhs,
            &mut lhs_i,
            &mut rhs_i,
            &mut packed_result,
            &mut i_packed_result,
            &mut msb_doc_id_groups_result,
            &mut i_msb_doc_id_groups_result,
        );

        let packed_result_ptr = Box::into_raw(packed_result) as *mut _;
        let msb_doc_id_groups_result_ptr = Box::into_raw(msb_doc_id_groups_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(packed_result_ptr, i_packed_result, min),
                if FIRST {
                    Vec::from_raw_parts(
                        msb_doc_id_groups_result_ptr,
                        i_msb_doc_id_groups_result,
                        lhs.packed.len() + 1,
                    )
                } else {
                    Vec::from_raw_parts(msb_doc_id_groups_result_ptr, 0, 0)
                },
            )
        }
    }

    fn inner_intersect<const FIRST: bool, L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>]>,
        i_packed_result: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>]>,
        i_msb_doc_id_groups_result: &mut usize
    );

    fn intersection_buffer_size<L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,
    ) -> usize {
        lhs.packed.len().min(rhs.packed.len())
    }
}
