use std::mem::MaybeUninit;

use crate::roaringish::{
    get_values, BorrowRoaringishPacked, Packed, ADD_ONE_GROUP, MASK_DOC_ID_GROUP, MASK_VALUES,
};

use super::{private::IntersectSeal, Intersect};

pub struct NaiveIntersect;
impl IntersectSeal for NaiveIntersect {}

impl Intersect for NaiveIntersect {
    #[inline(always)]
    fn inner_intersect<const FIRST: bool, L: Packed, R: Packed>(
        lhs: &BorrowRoaringishPacked<L>,
        rhs: &BorrowRoaringishPacked<R>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>]>,
        i_packed_result: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>]>,
        i_msb_doc_id_groups_result: &mut usize,
    ) {
        while *lhs_i < lhs.packed.len() && *rhs_i < rhs.packed.len() {
            let lhs_packed: u64 = unsafe { (*lhs.packed.get_unchecked(*lhs_i)).into() };
            let rhs_packed: u64 = unsafe { (*rhs.packed.get_unchecked(*rhs_i)).into() };

            let lhs_doc_id_groups = lhs_packed & MASK_DOC_ID_GROUP;
            let rhs_doc_id_groups = rhs_packed & MASK_DOC_ID_GROUP;

            let lhs_values = get_values(lhs_packed);
            let rhs_values = get_values(rhs_packed);

            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    if FIRST {
                        packed_result
                            .get_unchecked_mut(*i_packed_result)
                            .write(lhs_doc_id_groups | ((lhs_values << 1) & rhs_values) as u64);
                        msb_doc_id_groups_result
                            .get_unchecked_mut(*i_msb_doc_id_groups_result)
                            .write(lhs_doc_id_groups + ADD_ONE_GROUP);
                        *i_msb_doc_id_groups_result += (lhs_values & 0x8000 > 1) as usize;
                    } else {
                        packed_result
                            .get_unchecked_mut(*i_packed_result)
                            .write(lhs_doc_id_groups | (1 & rhs_values) as u64);
                    }
                }
                *i_packed_result += 1;
                *lhs_i += 1;
                *rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                *rhs_i += 1;
            } else {
                if FIRST {
                    unsafe {
                        msb_doc_id_groups_result
                            .get_unchecked_mut(*i_msb_doc_id_groups_result)
                            .write(lhs_doc_id_groups + ADD_ONE_GROUP);
                        *i_msb_doc_id_groups_result += (lhs_values & 0x8000 > 1) as usize;
                    }
                }
                *lhs_i += 1;
            }
        }
    }
}
