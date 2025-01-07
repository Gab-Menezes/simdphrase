use std::{mem::MaybeUninit, sync::atomic::Ordering::Relaxed};

use crate::{
    roaringish::{clear_values, unpack_values, Aligned, ADD_ONE_GROUP},
    Aligned64, BorrowRoaringishPacked, Stats,
};

use super::{private::IntersectSeal, Intersect};

#[inline(always)]
fn check_msb(
    packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
    i: &mut usize,

    doc_id_group: u64,
    values: u16,
    change_doc_id_group: impl FnOnce(u64, u64) -> u64,

    lhs_len: u16,
    check_mask: u16,
    and_mask: u16,
    rotate: impl FnOnce(u16, u32) -> u16,

    other: &[u64],

    k: usize,
    f: impl FnOnce(usize) -> usize,
) {
    if values & check_mask == 0 {
        return;
    }

    let Some(packed) = other.get(f(k)) else {
        return;
    };

    let packed_doc_id_group = clear_values(*packed);
    let packed_values = unpack_values(*packed);
    if change_doc_id_group(doc_id_group, ADD_ONE_GROUP) != packed_doc_id_group {
        return;
    }

    let intersection = rotate(values, lhs_len as u32) & and_mask & packed_values;

    check_intersection(packed_result, i, doc_id_group.max(packed_doc_id_group), intersection);
}

#[inline(always)]
fn check_intersection(
    packed_result: &mut Box<[MaybeUninit<u64>], Aligned64>,
    i: &mut usize,
    doc_id_group: u64,
    intersection: u16,
) {
    if intersection == 0 {
        return;
    }
    unsafe {
        packed_result
            .get_unchecked_mut(*i)
            .write(doc_id_group | intersection as u64);
    }
    *i += 1;
}

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

        if lhs.len() <= rhs.len() {
            let mut rhs = rhs.0;
            for lhs_packed in lhs.0.into_iter().copied() {
                let lhs_doc_id_group = clear_values(lhs_packed);
                let lhs_values = unpack_values(lhs_packed);
                let k = match rhs
                    .binary_search_by_key(&lhs_doc_id_group, |rhs_packed| clear_values(*rhs_packed))
                {
                    Ok(k) => {
                        let rhs_packed = unsafe { *rhs.get_unchecked(k) };
                        let rhs_values = unpack_values(rhs_packed);
                        let intersection = (lhs_values << lhs_len) & rhs_values;

                        let packed_result_init = unsafe {
                            MaybeUninit::slice_assume_init_mut(
                                packed_result.get_unchecked_mut(..*i),
                            )
                        };
                        match packed_result_init.last_mut() {
                            Some(last_packed_result) => {
                                let last_packed_result_doc_id_group =
                                    clear_values(*last_packed_result);

                                if last_packed_result_doc_id_group == lhs_doc_id_group {
                                    *last_packed_result |= intersection as u64;
                                } else {
                                    check_intersection(
                                        packed_result,
                                        i,
                                        lhs_doc_id_group,
                                        intersection,
                                    );
                                }

                                check_msb(
                                    packed_result,
                                    i,
                                    lhs_doc_id_group,
                                    lhs_values,
                                    |a, b| a + b,
                                    lhs_len,
                                    msb_mask,
                                    lsb_mask,
                                    |values, lhs_len| values.rotate_left(lhs_len),
                                    rhs,
                                    k,
                                    |k| k + 1,
                                );
                            }
                            None => {
                                check_intersection(
                                    packed_result,
                                    i,
                                    lhs_doc_id_group,
                                    intersection,
                                );

                                check_msb(
                                    packed_result,
                                    i,
                                    lhs_doc_id_group,
                                    lhs_values,
                                    |a, b| a + b,
                                    lhs_len,
                                    msb_mask,
                                    lsb_mask,
                                    |values, lhs_len| values.rotate_left(lhs_len),
                                    rhs,
                                    k,
                                    |k| k + 1,
                                );
                            }
                        }
                        k
                    }
                    Err(k) => {
                        check_msb(
                            packed_result,
                            i,
                            lhs_doc_id_group,
                            lhs_values,
                            |a, b| a + b,
                            lhs_len,
                            msb_mask,
                            lsb_mask,
                            |values, lhs_len| values.rotate_left(lhs_len),
                            rhs,
                            k,
                            |k| k,
                        );
                        k
                    }
                };
                rhs = &rhs[k..];
            }
        } else {
            let mut lhs = lhs.0;
            for rhs_packed in rhs.0.into_iter().copied() {
                let rhs_doc_id_group = clear_values(rhs_packed);
                let rhs_values = unpack_values(rhs_packed);
                let k = match lhs
                    .binary_search_by_key(&rhs_doc_id_group, |lhs_packed| clear_values(*lhs_packed))
                {
                    Ok(k) => {
                        let lhs_packed = unsafe { *lhs.get_unchecked(k) };
                        let lhs_values = unpack_values(lhs_packed);
                        let intersection = (lhs_values << lhs_len) & rhs_values;
                        let packed_result_init = unsafe {
                            MaybeUninit::slice_assume_init_mut(
                                packed_result.get_unchecked_mut(..*i),
                            )
                        };
                        match packed_result_init.last_mut() {
                            Some(last_packed_result) => {
                                let last_packed_result_doc_id_group =
                                    clear_values(*last_packed_result);

                                if last_packed_result_doc_id_group == rhs_doc_id_group {
                                    *last_packed_result |= intersection as u64;
                                } else {
                                    check_intersection(
                                        packed_result,
                                        i,
                                        rhs_doc_id_group,
                                        intersection,
                                    );
                                }

                                check_msb(
                                    packed_result,
                                    i,
                                    rhs_doc_id_group,
                                    rhs_values,
                                    |a, b| a - b,
                                    lhs_len,
                                    lsb_mask,
                                    msb_mask,
                                    |values, lhs_len| values.rotate_right(lhs_len),
                                    lhs,
                                    k,
                                    |k| k + 1,
                                );
                            }
                            None => {
                                check_intersection(
                                    packed_result,
                                    i,
                                    rhs_doc_id_group,
                                    intersection,
                                );

                                check_msb(
                                    packed_result,
                                    i,
                                    rhs_doc_id_group,
                                    rhs_values,
                                    |a, b| a - b,
                                    lhs_len,
                                    lsb_mask,
                                    msb_mask,
                                    |values, lhs_len| values.rotate_right(lhs_len),
                                    lhs,
                                    k,
                                    |k| k + 1,
                                );
                            }
                        }

                        k
                    }
                    Err(k) => {
                        check_msb(
                            packed_result,
                            i,
                            rhs_doc_id_group,
                            rhs_values,
                            |a, b| a - b,
                            lhs_len,
                            lsb_mask,
                            msb_mask,
                            |values, lhs_len| values.rotate_right(lhs_len),
                            lhs,
                            k,
                            |k| k.saturating_sub(1),
                        );
                        k
                    }
                };
                lhs = &lhs[k..];
            }
        }

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
