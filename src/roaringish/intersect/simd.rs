use std::{
    mem::MaybeUninit,
    simd::{cmp::SimdPartialEq, Mask, Simd},
};
use crate::roaringish::BorrowRoaringishPacked;

use super::{private::IntersectSeal, Intersect};

#[inline(always)]
fn word_wise_check<F: Fn(u64) -> u64>(
    a: &[u64],
    lhs_i: usize,
    b: &[u64],
    rhs_i: usize,
    op: F,
) -> Mask<i32, 16> {
    unsafe {
        let a0 = op(*a.get_unchecked(lhs_i + 0)) as u32;
        let a1 = op(*a.get_unchecked(lhs_i + 1)) as u32;
        let a2 = op(*a.get_unchecked(lhs_i + 2)) as u32;
        let a3 = op(*a.get_unchecked(lhs_i + 3)) as u32;
        let amask = Simd::from_array([
            a0, a0, a0, a0, a1, a1, a1, a1, a2, a2, a2, a2, a3, a3, a3, a3,
        ]);

        let b0 = op(*b.get_unchecked(rhs_i + 0)) as u32;
        let b1 = op(*b.get_unchecked(rhs_i + 1)) as u32;
        let b2 = op(*b.get_unchecked(rhs_i + 2)) as u32;
        let b3 = op(*b.get_unchecked(rhs_i + 3)) as u32;
        let bmask = Simd::from_array([
            b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3,
        ]);

        amask.simd_eq(bmask)
    }
}

#[inline(always)]
fn cmovnzu16(value: &mut MaybeUninit<u16>, v: &u16, condition: u8) {
    unsafe {
        std::arch::asm! {
            "test {0}, {0}",
            "cmovnz {1:r}, {2:r}",
            in(reg_byte) condition,
            inlateout(reg) *value,
            in(reg) *v,
            options(pure, nomem, nostack),
        };
    }
}

#[inline(always)]
fn cmovnzu64(value: &mut MaybeUninit<u64>, v: &u64, condition: u8) {
    unsafe {
        std::arch::asm! {
            "test {0}, {0}",
            "cmovnz {1:r}, {2:r}",
            in(reg_byte) condition,
            inlateout(reg) *value,
            in(reg) *v,
            options(pure, nomem, nostack),
        };
    }
}

#[inline(always)]
fn get_b_pos<const SHIFT: usize>(bitmask: u64, b_positions: &[u16], rhs_i: usize) -> &u16 {
    let idx = (bitmask >> SHIFT).trailing_zeros() % 4;
    unsafe { b_positions.get_unchecked(rhs_i + idx as usize) }
}

pub struct SimdIntersectCMOV;
impl IntersectSeal for SimdIntersectCMOV {}

impl Intersect for SimdIntersectCMOV {
    fn intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> (Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>) {
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

        let end_lhs = lhs_doc_id_groups.len() / 4 * 4;
        let end_rhs = rhs_doc_id_groups.len() / 4 * 4;
        let a = unsafe { lhs_doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs_doc_id_groups.get_unchecked(..end_rhs) };
        let a_positions = unsafe { lhs_positions.get_unchecked(..end_lhs) };
        let b_positions = unsafe { rhs_positions.get_unchecked(..end_rhs) };
        while lhs_i < a.len() && rhs_i < b.len() {
            let lsb = word_wise_check(a, lhs_i, b, rhs_i, |v| v).to_bitmask();
            if lsb >= 1 {
                // this if is very likely to happend

                let msb = word_wise_check(a, lhs_i, b, rhs_i, |v| (v & 0xFFFFFFFF00000000) >> 32)
                    .to_bitmask();
                // a0, a0, a0, a0 | a1, a1, a1, a1 | a2, a2, a2, a2 | a3, a3, a3, a3
                // b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3
                let bitmask = lsb & msb;

                let a0 = unsafe { a.get_unchecked(lhs_i + 0) };
                let a0_pos = unsafe { a_positions.get_unchecked(lhs_i + 0) };
                let a1 = unsafe { a.get_unchecked(lhs_i + 1) };
                let a1_pos = unsafe { a_positions.get_unchecked(lhs_i + 1) };
                let a2 = unsafe { a.get_unchecked(lhs_i + 2) };
                let a2_pos = unsafe { a_positions.get_unchecked(lhs_i + 2) };
                let a3 = unsafe { a.get_unchecked(lhs_i + 3) };
                let a3_pos = unsafe { a_positions.get_unchecked(lhs_i + 3) };

                let b3 = unsafe { b.get_unchecked(rhs_i + 3) };

                let a0b0123 = ((bitmask & 0b0000_0000_0000_1111) > 0) as u8;
                let a1b0123 = ((bitmask & 0b0000_0000_1111_0000) > 0) as u8;
                let a2b0123 = ((bitmask & 0b0000_1111_0000_0000) > 0) as u8;
                let a3b0123 = ((bitmask & 0b1111_0000_0000_0000) > 0) as u8;

                cmovnzu64(
                    unsafe { doc_id_groups_result.get_unchecked_mut(i) },
                    a0,
                    a0b0123,
                );
                cmovnzu16(
                    unsafe { rhs_positions_result.get_unchecked_mut(i) },
                    get_b_pos::<0>(bitmask, b_positions, rhs_i),
                    a0b0123,
                );
                if FIRST {
                    cmovnzu16(
                        unsafe { lhs_positions_result.get_unchecked_mut(i) },
                        a0_pos,
                        a0b0123,
                    );
                }
                i += a0b0123 as usize;

                cmovnzu64(
                    unsafe { doc_id_groups_result.get_unchecked_mut(i) },
                    a1,
                    a1b0123,
                );
                cmovnzu16(
                    unsafe { rhs_positions_result.get_unchecked_mut(i) },
                    get_b_pos::<4>(bitmask, b_positions, rhs_i),
                    a1b0123,
                );
                if FIRST {
                    cmovnzu16(
                        unsafe { lhs_positions_result.get_unchecked_mut(i) },
                        a1_pos,
                        a1b0123,
                    );
                }
                i += a1b0123 as usize;

                cmovnzu64(
                    unsafe { doc_id_groups_result.get_unchecked_mut(i) },
                    a2,
                    a2b0123,
                );
                cmovnzu16(
                    unsafe { rhs_positions_result.get_unchecked_mut(i) },
                    get_b_pos::<8>(bitmask, b_positions, rhs_i),
                    a2b0123,
                );
                if FIRST {
                    cmovnzu16(
                        unsafe { lhs_positions_result.get_unchecked_mut(i) },
                        a2_pos,
                        a2b0123,
                    );
                }
                i += a2b0123 as usize;

                cmovnzu64(
                    unsafe { doc_id_groups_result.get_unchecked_mut(i) },
                    a3,
                    a3b0123,
                );
                cmovnzu16(
                    unsafe { rhs_positions_result.get_unchecked_mut(i) },
                    get_b_pos::<12>(bitmask, b_positions, rhs_i),
                    a3b0123,
                );
                if FIRST {
                    cmovnzu16(
                        unsafe { lhs_positions_result.get_unchecked_mut(i) },
                        a3_pos,
                        a3b0123,
                    );
                }
                i += a3b0123 as usize;

                // branching in this case seems to be the fastest
                match a3.cmp(b3) {
                    std::cmp::Ordering::Greater => rhs_i += 4,
                    std::cmp::Ordering::Less => lhs_i += 4,
                    std::cmp::Ordering::Equal => {
                        lhs_i += 4;
                        rhs_i += 4;
                    }
                }
            } else if unsafe { *a.get_unchecked(lhs_i + 3) > *b.get_unchecked(rhs_i + 3) } {
                // we want to avoid this else if from running, since it's almost
                // a 50-50%, making it hard to predict, for this reason we repeat
                // this same check inside the
                rhs_i += 4;
            } else {
                lhs_i += 4;
            }
        }

        // for the remaining elements we can do a 2 pointer approach
        // since the 2 slices will be small
        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { lhs_doc_id_groups.get_unchecked(lhs_i) };
            let rhs_doc_id_groups = unsafe { rhs_doc_id_groups.get_unchecked(rhs_i) };
            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(*lhs_doc_id_groups)
                };
                if FIRST {
                    let lhs = unsafe { *lhs_positions.get_unchecked(lhs_i) };
                    unsafe { lhs_positions_result.get_unchecked_mut(i).write(lhs) };

                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                } else {
                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                }

                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                rhs_i += 1;
            } else {
                lhs_i += 1;
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

pub struct SimdIntersectBranch;
impl IntersectSeal for SimdIntersectBranch {}

impl Intersect for SimdIntersectBranch {
    fn intersect<const FIRST: bool>(
        lhs: &BorrowRoaringishPacked,
        rhs: &BorrowRoaringishPacked,
    ) -> (Vec<u64>, Vec<u16>, Vec<u16>, Vec<u64>) {
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

        let end_lhs = lhs_doc_id_groups.len() / 4 * 4;
        let end_rhs = rhs_doc_id_groups.len() / 4 * 4;
        let a = unsafe { lhs_doc_id_groups.get_unchecked(..end_lhs) };
        let b = unsafe { rhs_doc_id_groups.get_unchecked(..end_rhs) };
        let a_positions = unsafe { lhs_positions.get_unchecked(..end_lhs) };
        let b_positions = unsafe { rhs_positions.get_unchecked(..end_rhs) };
        while lhs_i < a.len() && rhs_i < b.len() {
            let lsb = word_wise_check(a, lhs_i, b, rhs_i, |v| v).to_bitmask();
            if lsb >= 1 {
                // this if is very likely to happend

                let msb = word_wise_check(a, lhs_i, b, rhs_i, |v| (v & 0xFFFFFFFF00000000) >> 32)
                    .to_bitmask();
                // a0, a0, a0, a0 | a1, a1, a1, a1 | a2, a2, a2, a2 | a3, a3, a3, a3
                // b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3
                let bitmask = lsb & msb;

                let a0 = unsafe { a.get_unchecked(lhs_i + 0) };
                let a0_pos = unsafe { a_positions.get_unchecked(lhs_i + 0) };
                let a1 = unsafe { a.get_unchecked(lhs_i + 1) };
                let a1_pos = unsafe { a_positions.get_unchecked(lhs_i + 1) };
                let a2 = unsafe { a.get_unchecked(lhs_i + 2) };
                let a2_pos = unsafe { a_positions.get_unchecked(lhs_i + 2) };
                let a3 = unsafe { a.get_unchecked(lhs_i + 3) };
                let a3_pos = unsafe { a_positions.get_unchecked(lhs_i + 3) };

                let b3 = unsafe { b.get_unchecked(rhs_i + 3) };

                let a0b0123 = (bitmask & 0b0000_0000_0000_1111) > 0;
                let a0b3 = (bitmask & 0b0000_0000_0000_1000) > 0;

                let a1b0123 = (bitmask & 0b0000_0000_1111_0000) > 0;
                let a1b3 = (bitmask & 0b0000_0000_1000_0000) > 0;

                let a2b0123 = (bitmask & 0b0000_1111_0000_0000) > 0;
                let a2b3 = (bitmask & 0b0000_1000_0000_0000) > 0;

                let a3b0123 = (bitmask & 0b1111_0000_0000_0000) > 0;
                let a3b3 = (bitmask & 0b1000_0000_0000_0000) > 0;

                if a0b0123 {
                    unsafe { doc_id_groups_result.get_unchecked_mut(i).write(*a0) };
                    unsafe {
                        rhs_positions_result
                            .get_unchecked_mut(i)
                            .write(*get_b_pos::<0>(bitmask, b_positions, rhs_i))
                    };
                    if FIRST {
                        unsafe { lhs_positions_result.get_unchecked_mut(i).write(*a0_pos) };
                    }
                    i += 1;
                    if a0b3 {
                        rhs_i += 4;
                        continue;
                    }
                }

                if a1b0123 {
                    unsafe { doc_id_groups_result.get_unchecked_mut(i).write(*a1) };
                    unsafe {
                        rhs_positions_result
                            .get_unchecked_mut(i)
                            .write(*get_b_pos::<4>(bitmask, b_positions, rhs_i))
                    };
                    if FIRST {
                        unsafe { lhs_positions_result.get_unchecked_mut(i).write(*a1_pos) };
                    }
                    i += 1;
                    if a1b3 {
                        rhs_i += 4;
                        continue;
                    }
                }

                if a2b0123 {
                    unsafe { doc_id_groups_result.get_unchecked_mut(i).write(*a2) };
                    unsafe {
                        rhs_positions_result
                            .get_unchecked_mut(i)
                            .write(*get_b_pos::<8>(bitmask, b_positions, rhs_i))
                    };
                    if FIRST {
                        unsafe { lhs_positions_result.get_unchecked_mut(i).write(*a2_pos) };
                    }
                    i += 1;
                    if a2b3 {
                        rhs_i += 4;
                        continue;
                    }
                }

                if a3b0123 {
                    unsafe { doc_id_groups_result.get_unchecked_mut(i).write(*a3) };
                    unsafe {
                        rhs_positions_result
                            .get_unchecked_mut(i)
                            .write(*get_b_pos::<12>(bitmask, b_positions, rhs_i))
                    };
                    if FIRST {
                        unsafe { lhs_positions_result.get_unchecked_mut(i).write(*a3_pos) };
                    }
                    i += 1;
                    lhs_i += 4;
                    if a3b3 {
                        rhs_i += 4;
                    }
                    continue;
                }

                if a3 > b3 {
                    rhs_i += 4;
                } else {
                    lhs_i += 4;
                }
            } else if unsafe { *a.get_unchecked(lhs_i + 3) > *b.get_unchecked(rhs_i + 3) } {
                // we want to avoid this else if from running, since it's almost
                // a 50-50%, making it hard to predict, for this reason we repeat
                // this same check inside the
                rhs_i += 4;
            } else {
                lhs_i += 4;
            }
        }

        // for the remaining elements we can do a 2 pointer approach
        // since the 2 slices will be small
        while lhs_i < lhs_doc_id_groups.len() && rhs_i < rhs_doc_id_groups.len() {
            let lhs_doc_id_groups = unsafe { lhs_doc_id_groups.get_unchecked(lhs_i) };
            let rhs_doc_id_groups = unsafe { rhs_doc_id_groups.get_unchecked(rhs_i) };
            if lhs_doc_id_groups == rhs_doc_id_groups {
                unsafe {
                    doc_id_groups_result
                        .get_unchecked_mut(i)
                        .write(*lhs_doc_id_groups)
                };
                if FIRST {
                    let lhs = unsafe { *lhs_positions.get_unchecked(lhs_i) };
                    unsafe { lhs_positions_result.get_unchecked_mut(i).write(lhs) };

                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                } else {
                    let rhs = unsafe { *rhs_positions.get_unchecked(rhs_i) };
                    unsafe { rhs_positions_result.get_unchecked_mut(i).write(rhs) };
                }

                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            } else if lhs_doc_id_groups > rhs_doc_id_groups {
                rhs_i += 1;
            } else {
                lhs_i += 1;
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
