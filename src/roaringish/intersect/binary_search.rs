use std::{mem::MaybeUninit, sync::atomic::Ordering::Relaxed};

use crate::{
    roaringish::{clear_values, unpack_values, Aligned, ADD_ONE_GROUP},
    Aligned64, BorrowRoaringishPacked, Stats,
};

use super::{private::IntersectSeal, Intersect};

pub struct BinarySearchIntersect;
impl IntersectSeal for BinarySearchIntersect {}
impl Intersect for BinarySearchIntersect {
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
    ) {
        let b = std::time::Instant::now();

        // if lhs.len() <= rhs.len() {
            let mut rhs = rhs.0;
            for lhs_packed in lhs.0.into_iter().copied() {
                let lhs_doc_id_group = clear_values(lhs_packed);
                let lhs_values = unpack_values(lhs_packed);
                match rhs.binary_search_by_key(&lhs_doc_id_group, |rhs_packed| clear_values(*rhs_packed))
                {
                    Ok(k) => {
                        let rhs_packed = unsafe { *rhs.get_unchecked(k) };
                        let rhs_values = unpack_values(rhs_packed);
                        let intersection = (lhs_values << lhs_len) & rhs_values;

                        let packed_result_init = unsafe { MaybeUninit::slice_assume_init_mut(packed_result.get_unchecked_mut(..*i)) };
                        match packed_result_init.last_mut() {
                            Some(last_packed_result) => {
                                let last_packed_result_id_group = clear_values(*last_packed_result);
                                if intersection > 0 {
                                    if last_packed_result_id_group == lhs_doc_id_group {
                                        *last_packed_result |= intersection as u64;
                                    } else {
                                        unsafe {
                                            packed_result
                                            .get_unchecked_mut(*i)
                                            .write(lhs_doc_id_group | intersection as u64);
                                        }
                                        *i += 1;
                                    }
                                }

                                if lhs_values & msb_mask > 0 {
                                    let Some(last_packed_result) = rhs.get(k + 1) else {
                                        rhs = &rhs[k..];
                                        continue;
                                    };

                                    let last_packed_result_id_group = clear_values(*last_packed_result);
                                    let last_packed_result_values = unpack_values(*last_packed_result);
                                    if last_packed_result_id_group != lhs_doc_id_group + ADD_ONE_GROUP {
                                        rhs = &rhs[k..];
                                        continue;
                                    }

                                    let intersection = lhs_values.rotate_left(lhs_len as u32) & lsb_mask & last_packed_result_values;
                                    if intersection > 0 {
                                        unsafe {
                                            packed_result
                                            .get_unchecked_mut(*i)
                                            .write(last_packed_result_id_group | intersection as u64);
                                        }
                                        *i += 1;
                                    }
                                }
                            },
                            None => {
                                if intersection > 0 {
                                    unsafe {
                                        packed_result
                                        .get_unchecked_mut(*i)
                                        .write(lhs_doc_id_group | intersection as u64);
                                    }
                                    *i += 1;
                                }

                                if lhs_values & msb_mask > 0 {
                                    let Some(last_packed_result) = rhs.get(k + 1) else {
                                        rhs = &rhs[k..];
                                        continue;
                                    };

                                    let last_packed_result_id_group = clear_values(*last_packed_result);
                                    let last_packed_result_values = unpack_values(*last_packed_result);
                                    if last_packed_result_id_group != lhs_doc_id_group + ADD_ONE_GROUP {
                                        rhs = &rhs[k..];
                                        continue;
                                    }

                                    let intersection = lhs_values.rotate_left(lhs_len as u32) & lsb_mask & last_packed_result_values;
                                    if intersection > 0 {
                                        unsafe {
                                            packed_result
                                            .get_unchecked_mut(*i)
                                            .write(last_packed_result_id_group | intersection as u64);
                                        }
                                        *i += 1;
                                    }
                                }
                            },
                        }
                        rhs = &rhs[k..];
                    },
                    Err(k) => {
                        if lhs_values & msb_mask > 0 {
                            let Some(last_packed_result) = rhs.get(k) else {
                                rhs = &rhs[k..];
                                continue;
                            };

                            let last_packed_result_id_group = clear_values(*last_packed_result);
                            let last_packed_result_values = unpack_values(*last_packed_result);
                            if last_packed_result_id_group != lhs_doc_id_group + ADD_ONE_GROUP {
                                rhs = &rhs[k..];
                                continue;
                            }

                            let intersection = lhs_values.rotate_left(lhs_len as u32) & lsb_mask & last_packed_result_values;
                            if intersection > 0 {
                                unsafe {
                                    packed_result
                                    .get_unchecked_mut(*i)
                                    .write(last_packed_result_id_group | intersection as u64);
                                }
                                *i += 1;
                            }
                        }
                        rhs = &rhs[k..];
                    },
                }
            }
        // } else {
        // }
    
        if FIRST {
            stats
            .first_intersect_binary
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
        } else {
            stats
            .second_intersect_binary
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
        }
    }

    fn intersection_buffer_size(
        lhs: BorrowRoaringishPacked<'_, Aligned>,
        rhs: BorrowRoaringishPacked<'_, Aligned>,
    ) -> usize {
        2 * lhs.0.len().min(rhs.0.len())
    }

    fn needs_second_pass() -> bool {
        false
    }
}
