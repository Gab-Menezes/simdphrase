use std::mem::MaybeUninit;

use crate::{allocator::Aligned64, roaringish::{Aligned, BorrowRoaringishPacked}};

use super::{private::IntersectSeal, Intersect};

pub struct NaiveIntersect;
impl IntersectSeal for NaiveIntersect {}

impl Intersect for NaiveIntersect {
    fn inner_intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        i: &mut usize,

        msb_doc_id_groups_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        msb_values_result: &mut Box<[MaybeUninit<u16>], Aligned64>,
        j: &mut usize,

        lhs_len: u16,
        msb_mask: u16,
        lsb_mask: u16
    ) {
        while *lhs_i < lhs.doc_id_groups.len() && *rhs_i < rhs.doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { *lhs.doc_id_groups.get_unchecked(*lhs_i) };
            let rhs_doc_id_groups = unsafe { *rhs.doc_id_groups.get_unchecked(*rhs_i) };

            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(*i)
                        .write(lhs_doc_id_groups);

                    let lhs_values = *lhs.values.get_unchecked(*lhs_i);
                    let rhs_values = *rhs.values.get_unchecked(*rhs_i);
                    if FIRST {
                        values_result.get_unchecked_mut(*i).write((lhs_values << lhs_len) & rhs_values);

                        // TODO: this sucks
                        msb_doc_id_groups_result
                            .get_unchecked_mut(*j)
                            .write(lhs_doc_id_groups + 1);
                        msb_values_result
                            .get_unchecked_mut(*j)
                            .write(lhs_values);
                        *j += (lhs_values & msb_mask > 1) as usize;
                    } else {
                        values_result.get_unchecked_mut(*i)
                        .write(
                            (lhs_values.rotate_left(lhs_len as u32) & lsb_mask) & rhs_values
                        );
                    }
                }
                *i += 1;
                *lhs_i += 1;
                *rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                *rhs_i += 1;
            } else {
                if FIRST {
                    unsafe {
                        let lhs_values = *lhs.values.get_unchecked(*lhs_i);
                        msb_doc_id_groups_result
                            .get_unchecked_mut(*j)
                            .write(lhs_doc_id_groups + 1);
                        msb_values_result
                            .get_unchecked_mut(*j)
                            .write(lhs_values);
                        *j += (lhs_values & msb_mask > 1) as usize;
                    }
                }
                *lhs_i += 1;
            }
        }
    }

    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize {
        lhs.doc_id_groups.len().min(rhs.doc_id_groups.len())
    }
}
