use std::mem::MaybeUninit;

use crate::roaringish::{BorrowRoaringishPacked, Packed};

use super::{private::IntersectSeal, Intersect};

pub struct NaiveIntersect;
impl IntersectSeal for NaiveIntersect {}

impl Intersect for NaiveIntersect {
    fn intersect<const FIRST: bool, L: Packed, R: Packed>(lhs: &BorrowRoaringishPacked<L>, rhs: &BorrowRoaringishPacked<R>) -> (Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>) {
        let lhs_positions = lhs.positions;
        let rhs_positions = rhs.positions;
        let lhs_doc_id_groups = lhs.doc_id_groups;
        let rhs_doc_id_groups = rhs.doc_id_groups;

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let min = lhs_doc_id_groups.len().min(rhs_doc_id_groups.len());
        let mut i = 0;
        let mut j = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
        let mut msb_doc_id_groups_result: Box<[MaybeUninit<u64>]> = if FIRST {
            Box::new_uninit_slice(lhs_doc_id_groups.len() + 1)
        } else {
            Box::new_uninit_slice(0)
        };
        let mut lhs_positions_result: Box<[MaybeUninit<u16>]> = if FIRST {
            Box::new_uninit_slice(min)
        } else {
            Box::new_uninit_slice(0)
        };
        let mut rhs_positions_result: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(min);

        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { (*lhs_doc_id_groups.get_unchecked(lhs_i)).into() };
            let rhs_doc_id_groups = unsafe { (*rhs_doc_id_groups.get_unchecked(rhs_i)).into() };

            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(lhs_doc_id_groups);
                    let rhs = (*rhs_positions.get_unchecked(rhs_i)).into();
                    rhs_positions_result.get_unchecked_mut(i).write(rhs);
                    if FIRST {
                        let lhs = (*lhs_positions.get_unchecked(lhs_i)).into();
                        lhs_positions_result.get_unchecked_mut(i).write(lhs);

                        msb_doc_id_groups_result.get_unchecked_mut(j).write(lhs_doc_id_groups + 1);
                        j += (lhs & 0x8000 > 1) as usize;
                    }
                }

                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                rhs_i += 1;
            } else {
                if FIRST {
                    unsafe {
                        let lhs = (*lhs_positions.get_unchecked(lhs_i)).into();
                        msb_doc_id_groups_result.get_unchecked_mut(j).write(lhs_doc_id_groups + 1);
                        j += (lhs & 0x8000 > 1) as usize;
                    }
                }

                lhs_i += 1;
            }
        }

        let doc_id_groups_result_ptr = Box::into_raw(doc_id_groups_result) as *mut _;
        let lhs_positions_result_ptr = Box::into_raw(lhs_positions_result) as *mut _;
        let rhs_positions_result_ptr = Box::into_raw(rhs_positions_result) as *mut _;
        let msb_doc_id_groups_result_ptr = Box::into_raw(msb_doc_id_groups_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(doc_id_groups_result_ptr, i, min),

                if FIRST {
                    Vec::from_raw_parts(lhs_positions_result_ptr, i, min)
                } else {
                    Vec::from_raw_parts(lhs_positions_result_ptr, 0, 0)
                },

                Vec::from_raw_parts(rhs_positions_result_ptr, i, min),

                if FIRST {
                    Vec::from_raw_parts(msb_doc_id_groups_result_ptr, j, lhs_doc_id_groups.len() + 1)
                } else {
                    Vec::from_raw_parts(msb_doc_id_groups_result_ptr, 0, 0)
                },
            )
        }
    }
}