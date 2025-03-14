use std::{mem::MaybeUninit, sync::atomic::Ordering::Relaxed};

use crate::{
    Aligned64, BorrowRoaringishPacked, Stats,
    roaringish::{ADD_ONE_GROUP, Aligned, clear_values, unpack_values},
};

use super::{Intersect, Intersection, private::IntersectSeal};

pub struct GallopIntersectFirst;
impl IntersectSeal for GallopIntersectFirst {}
impl Intersection for GallopIntersectFirst {}

impl Intersect for GallopIntersectFirst {
    fn inner_intersect<const FIRST: bool>(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,

        lhs_i: &mut usize,
        rhs_i: &mut usize,

        packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        i: &mut usize,

        _msb_packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
        _j: &mut usize,

        add_to_group: u64,
        lhs_len: u16,
        _msb_mask: u16,
        lsb_mask: u16,

        stats: &Stats,
    ) {
        let b = std::time::Instant::now();

        while *lhs_i < lhs.len() && *rhs_i < rhs.len() {
            let mut lhs_delta = 1;
            let mut rhs_delta = 1;

            while *lhs_i < lhs.len()
                && clear_values(lhs[*lhs_i]) + add_to_group + if FIRST { 0 } else { ADD_ONE_GROUP }
                    < clear_values(rhs[*rhs_i])
            {
                *lhs_i += lhs_delta;
                lhs_delta *= 2;
            }
            *lhs_i -= lhs_delta / 2;

            while *rhs_i < rhs.len()
                && clear_values(rhs[*rhs_i])
                    < unsafe { clear_values(*lhs.get_unchecked(*lhs_i)) }
                        + add_to_group
                        + if FIRST { 0 } else { ADD_ONE_GROUP }
            {
                *rhs_i += rhs_delta;
                rhs_delta *= 2;
            }
            *rhs_i -= rhs_delta / 2;

            let lhs_packed = unsafe { *lhs.get_unchecked(*lhs_i) }
                + add_to_group
                + if FIRST { 0 } else { ADD_ONE_GROUP };
            let rhs_packed = unsafe { *rhs.get_unchecked(*rhs_i) };

            let lhs_doc_id_group = clear_values(lhs_packed);
            let rhs_doc_id_group = clear_values(rhs_packed);

            let lhs_values = unpack_values(lhs_packed);
            let rhs_values = unpack_values(rhs_packed);

            match lhs_doc_id_group.cmp(&rhs_doc_id_group) {
                std::cmp::Ordering::Less => *lhs_i += 1,
                std::cmp::Ordering::Greater => *rhs_i += 1,
                std::cmp::Ordering::Equal => {
                    let intersection = if FIRST {
                        (lhs_values << lhs_len) & rhs_values
                    } else {
                        lhs_values.rotate_left(lhs_len as u32) & lsb_mask & rhs_values
                    };
                    unsafe {
                        packed_result
                            .get_unchecked_mut(*i)
                            .write(lhs_doc_id_group | intersection as u64);
                    }
                    *i += (intersection > 0) as usize;

                    *lhs_i += 1;
                    *rhs_i += 1;
                }
            }
        }

        stats
            .first_intersect_gallop
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
    }

    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize {
        lhs.0.len().min(rhs.0.len())
    }
}
