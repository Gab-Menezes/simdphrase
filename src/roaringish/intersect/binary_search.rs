use std::{mem::MaybeUninit, sync::atomic::Ordering::Relaxed};

use crate::{
    roaringish::{clear_values, unpack_values, Aligned, ADD_ONE_GROUP},
    Aligned64, BorrowRoaringishPacked, Stats,
};

use super::{private::IntersectSeal, Intersect};


fn lhs_intersection(
    lhs_values: u16, 
    lhs_len: u32,
    lsb_mask: u16,
    rhs_values: u16
) -> u16 {
    lhs_values.rotate_left(lhs_len) & lsb_mask & rhs_values
}

fn lhs_choose_doc_id_group(_lhs_doc_id_group: u64, packed_doc_id_group: u64) -> u64 {
    packed_doc_id_group
}

fn rhs_intersection(
    rhs_values: u16, 
    lhs_len: u32,
    lsb_mask: u16,
    lhs_values: u16
) -> u16 {
    lhs_values.rotate_left(lhs_len) & lsb_mask & rhs_values
}

fn rhs_choose_doc_id_group(rhs_doc_id_group: u64, _packed_doc_id_group: u64) -> u64 {
    rhs_doc_id_group
}

macro_rules! check_msb {
    (
        $packed_result:ident,
        $i:ident,

        $doc_id_group:ident,
        $doc_id_group_expr:expr,
        $values:ident,
        
        $lhs_len:ident,
        $check_mask:ident,
        $and_mask:ident,
        $intersection:ident,

        $other:ident,

        $k:expr,

        $choose_doc_id_group:ident
    ) => {
        if $values & $check_mask == 0 {
            continue;
        }
    
        let Some(packed) = $other.get($k) else {
            continue;
        };
    
        let packed_doc_id_group = clear_values(*packed);
        let packed_values = unpack_values(*packed);
        if $doc_id_group_expr != packed_doc_id_group {
            continue;
        }
        
        check_intersection($packed_result, $i, $choose_doc_id_group($doc_id_group, packed_doc_id_group), $intersection($values, $lhs_len as u32, $and_mask, packed_values));
    };
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

                                check_msb!(
                                    packed_result,
                                    i,

                                    lhs_doc_id_group,
                                    lhs_doc_id_group + ADD_ONE_GROUP,
                                    lhs_values,

                                    lhs_len,
                                    msb_mask,
                                    lsb_mask,
                                    lhs_intersection,

                                    rhs,

                                    k + 1,

                                    lhs_choose_doc_id_group
                                );
                            }
                            None => {
                                check_intersection(
                                    packed_result,
                                    i,
                                    lhs_doc_id_group,
                                    intersection,
                                );

                                check_msb!(
                                    packed_result,
                                    i,

                                    lhs_doc_id_group,
                                    lhs_doc_id_group + ADD_ONE_GROUP,
                                    lhs_values,

                                    lhs_len,
                                    msb_mask,
                                    lsb_mask,
                                    lhs_intersection,

                                    rhs,

                                    k + 1,

                                    lhs_choose_doc_id_group
                                );
                            }
                        }
                        k
                    }
                    Err(k) => {
                        check_msb!(
                            packed_result,
                            i,

                            lhs_doc_id_group,
                            lhs_doc_id_group + ADD_ONE_GROUP,
                            lhs_values,

                            lhs_len,
                            msb_mask,
                            lsb_mask,
                            lhs_intersection,

                            rhs,

                            k,

                            lhs_choose_doc_id_group
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
                if rhs_packed >> 32 == 1469962 {
                    println!("rhs: {rhs_packed}: {:032b} {:016b} {rhs_values:016b}", rhs_packed >> 32, (rhs_packed & 0xFFFF0000) >> 16);
                }
                let k = match lhs
                    .binary_search_by_key(&rhs_doc_id_group, |lhs_packed| clear_values(*lhs_packed))
                {
                    Ok(k) => {
                        let lhs_packed = unsafe { *lhs.get_unchecked(k) };
                        let lhs_values = unpack_values(lhs_packed);
                        println!("lhs: {lhs_packed}: {:032b} {:016b} {lhs_values:016b}", lhs_packed >> 32, (lhs_packed & 0xFFFF0000) >> 16);
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

                                check_msb!(
                                    packed_result,
                                    i,
        
                                    rhs_doc_id_group,
                                    rhs_doc_id_group - ADD_ONE_GROUP,
                                    rhs_values,
        
                                    lhs_len,
                                    lsb_mask,
                                    lsb_mask,
                                    rhs_intersection,
        
                                    lhs,
        
                                    k.saturating_sub(1),
        
                                    rhs_choose_doc_id_group
                                );
                            }
                            None => {
                                check_intersection(
                                    packed_result,
                                    i,
                                    rhs_doc_id_group,
                                    intersection,
                                );

                                check_msb!(
                                    packed_result,
                                    i,
        
                                    rhs_doc_id_group,
                                    rhs_doc_id_group - ADD_ONE_GROUP,
                                    rhs_values,
        
                                    lhs_len,
                                    lsb_mask,
                                    lsb_mask,
                                    rhs_intersection,
        
                                    lhs,
        
                                    k.saturating_sub(1),
        
                                    rhs_choose_doc_id_group
                                );
                            }
                        }

                        k
                    }
                    Err(k) => {
                        check_msb!(
                            packed_result,
                            i,

                            rhs_doc_id_group,
                            rhs_doc_id_group - ADD_ONE_GROUP,
                            rhs_values,

                            lhs_len,
                            lsb_mask,
                            lsb_mask,
                            rhs_intersection,

                            lhs,

                            k.saturating_sub(1),

                            rhs_choose_doc_id_group
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
