use std::mem::MaybeUninit;

use crate::roaringish::BorrowRoaringishPacked;

use super::{private::IntersectSeal, Intersect};

pub struct GallopIntersect;
impl IntersectSeal for GallopIntersect {}

impl Intersect for GallopIntersect {
    fn intersect<const FIRST: bool>(lhs: &BorrowRoaringishPacked, rhs: &BorrowRoaringishPacked) -> (Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>) {
        let lhs_positions = lhs.positions;
        let rhs_positions = rhs.positions;
        let lhs_doc_id_groups = lhs.doc_id_groups;
        let rhs_doc_id_groups = rhs.doc_id_groups;

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let min = lhs_doc_id_groups.len().min(rhs_doc_id_groups.len());
        let mut i = 0;
        let mut doc_id_groups_result: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
        let mut lhs_positions_result: Box<[MaybeUninit<u16>]> = if FIRST {
            Box::new_uninit_slice(min)
        } else {
            Box::new_uninit_slice(0)
        };
        let mut rhs_positions_result: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(min);

        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let mut lhs_delta = 1;
            let mut rhs_delta = 1;

            while lhs_i < lhs_doc_id_groups.len()
                && lhs_doc_id_groups[lhs_i] < rhs_doc_id_groups[rhs_i]
            {
                lhs_i += lhs_delta;
                lhs_delta *= 2;
            }
            lhs_i -= lhs_delta / 2;

            while rhs_i < rhs_doc_id_groups.len()
                && rhs_doc_id_groups[rhs_i] < unsafe { *lhs_doc_id_groups.get_unchecked(lhs_i) }
            {
                rhs_i += rhs_delta;
                rhs_delta *= 2;
            }
            rhs_i -= rhs_delta / 2;

            let lhs_doc_id_groups = unsafe { lhs_doc_id_groups.get_unchecked(lhs_i) };
            let rhs_doc_id_groups = unsafe { rhs_doc_id_groups.get_unchecked(rhs_i) };
            match lhs_doc_id_groups.cmp(rhs_doc_id_groups) {
                std::cmp::Ordering::Less => lhs_i += 1,
                std::cmp::Ordering::Greater => rhs_i += 1,
                std::cmp::Ordering::Equal => unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(*lhs_doc_id_groups);
                    if FIRST {
                        let lhs = *lhs_positions.get_unchecked(lhs_i);
                        lhs_positions_result.get_unchecked_mut(i).write(lhs);

                        let rhs = *rhs_positions.get_unchecked(rhs_i);
                        rhs_positions_result.get_unchecked_mut(i).write(rhs);
                    } else {
                        let rhs = *rhs_positions.get_unchecked(rhs_i);
                        rhs_positions_result.get_unchecked_mut(i).write(rhs);
                    }

                    i += 1;
                    lhs_i += 1;
                    rhs_i += 1;
                },
            }
        }

        let doc_id_groups_result_ptr = Box::into_raw(doc_id_groups_result) as *mut _;
        let lhs_positions_result_ptr = Box::into_raw(lhs_positions_result) as *mut _;
        let rhs_positions_result_ptr = Box::into_raw(rhs_positions_result) as *mut _;
        unsafe {
            (
                Vec::from_raw_parts(doc_id_groups_result_ptr, i, min),
                if FIRST {
                    Vec::from_raw_parts(lhs_positions_result_ptr, i, min)
                } else {
                    Vec::from_raw_parts(lhs_positions_result_ptr, 0, 0)
                },
                Vec::from_raw_parts(rhs_positions_result_ptr, i, min),
                // TODO: FIX
                Vec::new()
            )
        }
    }
}