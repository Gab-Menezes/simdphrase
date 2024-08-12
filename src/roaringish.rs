pub mod intersect;

use arbitrary::{Arbitrary, Unstructured};
use rkyv::{Archive, Deserialize, Serialize};
#[allow(unused_imports)]
use std::arch::x86_64::{__m256i, __m512i};
use std::simd::cmp::SimdPartialOrd;
use std::{
    fmt::{Binary, Debug, Display},
    intrinsics::assume,
    mem::MaybeUninit,
    ops::Add,
    simd::{cmp::SimdPartialEq, Mask, Simd},
    sync::atomic::Ordering::Relaxed,
};

use crate::{
    pl::{ArchivedPostingList, PostingList},
    Stats,
};

use self::intersect::Intersect;

// use self::intersect::Intersect;

#[derive(Default, Serialize, Deserialize, Archive)]
#[archive_attr(derive(Debug))]
pub struct Roaringish {
    pub(crate) inner: Vec<u32>,
}

impl Binary for Roaringish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for packed in self.inner.iter() {
            list.entry_with(|f| {
                f.write_fmt(format_args!(
                    "{:016b} {:016b}",
                    get_group(*packed),
                    get_encoded_positions(*packed)
                ))
            });
        }

        list.finish()
    }
}

impl Debug for Roaringish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for packed in self.inner.iter() {
            list.entry_with(|f| {
                f.debug_tuple("")
                    .field(&get_group(*packed))
                    .field_with(|f| {
                        let encoded_values = get_encoded_positions(*packed);
                        f.debug_list()
                            .entries(
                                (0..16u32)
                                    .filter_map(|i| ((encoded_values >> i) & 1 == 1).then_some(i)),
                            )
                            .finish()
                    })
                    .finish()
            });
        }
        list.finish()
    }
}

impl Display for Roaringish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let it = self
            .inner
            .iter()
            .map(|packed| {
                let group = get_group(*packed);
                let encoded_values = get_encoded_positions(*packed);
                let s = group * 16;
                (0..16u32)
                    .filter_map(move |i| ((encoded_values >> i) & 1 == 1).then_some(i))
                    .map(move |i| s + i)
            })
            .flatten();

        f.debug_list().entries(it).finish()
    }
}

pub const MAX_VALUE: u32 = 16u32 * u16::MAX as u32;

const fn group(val: u32) -> u32 {
    val / 16
}

const fn value(val: u32) -> u32 {
    val % 16
}

const fn gv(val: u32) -> (u32, u32) {
    (group(val), value(val))
}

const fn make_group(group: u32) -> u32 {
    group << 16
}

const fn get_group(packed: u32) -> u32 {
    packed >> 16
}

const fn make_position(value: u32) -> u32 {
    1 << value
}

const fn get_encoded_positions(packed: u32) -> u32 {
    packed & 0x0000FFFF
}

const fn make_doc_id(doc_id: u32) -> u64 {
    (doc_id as u64) << 16
}

const fn get_doc_id(packed: u64) -> u32 {
    (packed >> 16) as u32
}

const fn get_group_from_doc_id_group(packed: u64) -> u32 {
    (packed & 0x00000000FFFF) as u32
}

impl Roaringish {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_positions(mut pos: Vec<u32>) -> Self {
        pos.sort_unstable();
        Self::from_positions_sorted(pos)
    }

    pub fn from_positions_sorted(mut pos: Vec<u32>) -> Self {
        if pos.is_empty() {
            return Self::new();
        }

        let mut values: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(pos.len());
        unsafe {
            assume(values.len() == pos.len());
        }
        for (pos, v) in pos.iter_mut().zip(values.iter_mut()) {
            let (group, value) = gv(*pos);
            *pos = group;
            v.write(value);
        }
        let groups = pos;
        let values = unsafe { values.assume_init() };

        let mut inner = Vec::new();
        let mut b = 0;
        for groups in groups.chunk_by(|g0, g1| g0 == g1) {
            let values = unsafe { values.get_unchecked(b..(b + groups.len())) };
            b += groups.len();

            unsafe {
                assume(values.len() <= 16);
            }

            let mut packed = make_group(groups[0]);
            for value in values {
                packed |= make_position(*value);
            }

            inner.push(packed);
        }

        Self { inner }
    }
}

#[derive(Default, Debug)]
pub struct RoaringishPacked {
    pub doc_id_groups: Box<[u64]>,
    pub positions: Box<[u16]>,
}

pub struct BorrowRoaringishPacked<'a> {
    pub(crate) doc_id_groups: &'a [u64],
    pub(crate) positions: &'a [u16],
}

impl<'a> BorrowRoaringishPacked<'a> {
    pub fn new(r: &'a RoaringishPacked) -> Self {
        Self {
            doc_id_groups: &r.doc_id_groups,
            positions: &r.positions,
        }
    }

    pub fn from_raw(doc_id_groups: &'a [u64], positions: &'a [u16]) -> Self {
        Self {
            doc_id_groups,
            positions,
        }
    }
}

impl RoaringishPacked {
    pub fn get_doc_ids(&self) -> Vec<u32> {
        if self.doc_id_groups.len() == 0 {
            return Vec::new();
        }

        if self.doc_id_groups.len() == 1 {
            return vec![get_doc_id(self.doc_id_groups[0])];
        }

        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(self.doc_id_groups.len());
        let mut i = 0;

        for [packed0, packed1] in self.doc_id_groups.array_windows::<2>() {
            let doc_id0 = get_doc_id(*packed0);
            let doc_id1 = get_doc_id(*packed1);
            if doc_id0 != doc_id1 {
                unsafe { doc_ids.get_unchecked_mut(i).write(doc_id0) };
                i += 1;
            }
        }

        unsafe {
            doc_ids
                .get_unchecked_mut(i)
                .write(get_doc_id(*self.doc_id_groups.last().unwrap_unchecked()))
        };
        i += 1;

        unsafe {
            Vec::from_raw_parts(
                Box::into_raw(doc_ids) as *mut _,
                i,
                self.doc_id_groups.len(),
            )
        }
    }

    pub fn intersect<I: Intersect>(&self, rhs: &Self, rhs_len: u32, stats: &Stats) -> Self {
        let lhs = BorrowRoaringishPacked::new(self);
        let rhs = BorrowRoaringishPacked::new(rhs);
        let b = std::time::Instant::now();
        let (doc_id_groups, lhs_positions, rhs_positions, msb_doc_id_groups) = I::intersect::<true>(&lhs, &rhs);
        stats
            .first_gallop
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let msb_packed = BorrowRoaringishPacked::from_raw(&msb_doc_id_groups, &[]);
        let b = std::time::Instant::now();
        let (msb_doc_id_groups, _, msb_rhs_positions, _) = I::intersect::<false>(&msb_packed, &rhs);

        stats
            .second_gallop
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        unsafe {
            assume(doc_id_groups.len() == lhs_positions.len());
            assume(doc_id_groups.len() == rhs_positions.len());

            assume(msb_doc_id_groups.len() == msb_rhs_positions.len());
        };

        let b = std::time::Instant::now();
        let positions_intersect: Box<[u16]> = lhs_positions
            .iter()
            .zip(rhs_positions.iter())
            .map(|(lhs, rhs)| (lhs << 1) & rhs)
            .collect();
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let msb_positions_intersect: Box<[u16]> =
            msb_rhs_positions.iter().map(|rhs| 1 & rhs).collect();
        stats
            .second_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let capacity = positions_intersect.len() + msb_positions_intersect.len();
        let mut r_doc_id_groups = Box::new_uninit_slice(capacity);
        let mut r_positions = Box::new_uninit_slice(capacity);
        let mut r_i = 0;
        let mut j = 0;
        for i in 0..positions_intersect.len() {
            unsafe {
                let doc_id_group = *doc_id_groups.get_unchecked(i);
                let intersection = *positions_intersect.get_unchecked(i);

                while j < msb_positions_intersect.len() {
                    let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(j);
                    let msb_intersection = *msb_positions_intersect.get_unchecked(j);
                    j += 1;

                    if msb_doc_id_group >= doc_id_group {
                        j -= 1;
                        break;
                    }

                    if msb_intersection > 0 {
                        r_doc_id_groups
                            .get_unchecked_mut(r_i)
                            .write(msb_doc_id_group);
                        r_positions.get_unchecked_mut(r_i).write(msb_intersection);
                        r_i += 1;
                    }
                }

                let write = intersection > 0;
                if write {
                    r_doc_id_groups.get_unchecked_mut(r_i).write(doc_id_group);
                    r_positions.get_unchecked_mut(r_i).write(intersection);
                    r_i += 1;
                }

                {
                    if j >= msb_positions_intersect.len() {
                        continue;
                    }

                    let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(j);
                    let msb_intersection = *msb_positions_intersect.get_unchecked(j);
                    j += 1;
                    if msb_doc_id_group != doc_id_group {
                        j -= 1;
                        continue;
                    }

                    if write {
                        // in this case no bit was set in the intersection,
                        // so we can just or the new value with the previous one
                        let r = r_positions.get_unchecked_mut(r_i - 1).assume_init();
                        r_positions
                            .get_unchecked_mut(r_i - 1)
                            .write(r | msb_intersection);
                    } else if msb_intersection > 0 {
                        r_doc_id_groups
                            .get_unchecked_mut(r_i)
                            .write(msb_doc_id_group);
                        r_positions.get_unchecked_mut(r_i).write(msb_intersection);
                        r_i += 1;
                    }
                }
            }
        }
        stats
            .first_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        for i in j..msb_positions_intersect.len() {
            unsafe {
                let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(i);
                let msb_intersection = *msb_positions_intersect.get_unchecked(i);
                if msb_intersection > 0 {
                    r_doc_id_groups
                        .get_unchecked_mut(r_i)
                        .write(msb_doc_id_group);
                    r_positions.get_unchecked_mut(r_i).write(msb_intersection);
                    r_i += 1;
                }
            }
        }
        stats
            .second_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let packed = unsafe {
            let doc_id_groups =
                Vec::from_raw_parts(Box::into_raw(r_doc_id_groups) as *mut _, r_i, capacity)
                    .into_boxed_slice();
            let positions =
                Vec::from_raw_parts(Box::into_raw(r_positions) as *mut _, r_i, capacity)
                    .into_boxed_slice();
            Self {
                doc_id_groups,
                positions,
            }
        };
        if rhs_len > 1 {
            let b = std::time::Instant::now();
            let r = &packed + (rhs_len - 1);
            stats
                .add_rhs
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
            r
        } else {
            packed
        }
    }
}

impl Add<u32> for &RoaringishPacked {
    type Output = RoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        unsafe {
            assume(self.doc_id_groups.len() == self.positions.len());
        }

        let n = self.doc_id_groups.len() * 2;
        let mut doc_id_groups = Box::new_uninit_slice(n);
        let mut positions = Box::new_uninit_slice(n);
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;
        for (doc_id_group, values) in self.doc_id_groups.iter().zip(self.positions.iter()) {
            let new_positions = values.rotate_left(rhs as u32);
            let postions_current_group = new_positions & mask;
            let postions_new_group = new_positions & bits_to_check;
            if postions_current_group > 0 {
                unsafe {
                    doc_id_groups.get_unchecked_mut(i).write(*doc_id_group);
                    positions.get_unchecked_mut(i).write(postions_current_group);
                    i += 1;
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    doc_id_groups.get_unchecked_mut(i).write(*doc_id_group + 1);
                    positions.get_unchecked_mut(i).write(postions_new_group);
                    i += 1;
                }
            }
        }

        unsafe {
            RoaringishPacked {
                doc_id_groups: Vec::from_raw_parts(Box::into_raw(doc_id_groups) as *mut _, i, n)
                    .into_boxed_slice(),
                positions: Vec::from_raw_parts(Box::into_raw(positions) as *mut _, i, n)
                    .into_boxed_slice(),
            }
        }
    }
}

impl From<&PostingList> for RoaringishPacked {
    fn from(pl: &PostingList) -> Self {
        let mut doc_id_groups: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(pl.len_sum as usize);
        let mut positions: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(pl.len_sum as usize);

        let doc_ids = pl.doc_ids.as_slice();
        let roaringish = pl.positions.as_slice();
        unsafe { assume(doc_ids.len() == roaringish.len()) }

        let mut b = 0;
        for (doc_id, roaringish) in doc_ids.iter().zip(roaringish.iter()) {
            let doc_id = make_doc_id(*doc_id);

            let doc_id_groups =
                unsafe { doc_id_groups.get_unchecked_mut(b..(b + roaringish.inner.len())) };
            let positions = unsafe { positions.get_unchecked_mut(b..(b + roaringish.inner.len())) };
            b += roaringish.inner.len();

            for ((doc_id_group, position), packed) in doc_id_groups
                .iter_mut()
                .zip(positions.iter_mut())
                .zip(roaringish.inner.iter())
            {
                let group = get_group(*packed) as u64;
                let encoded_positions = get_encoded_positions(*packed) as u16;
                doc_id_group.write(doc_id | group);
                position.write(encoded_positions);
            }
        }

        let doc_id_groups = unsafe { doc_id_groups.assume_init() };
        let positions = unsafe { positions.assume_init() };
        Self {
            doc_id_groups,
            positions,
        }
    }
}

impl From<&ArchivedPostingList> for RoaringishPacked {
    fn from(pl: &ArchivedPostingList) -> Self {
        let mut doc_id_groups: Box<[MaybeUninit<u64>]> =
            Box::new_uninit_slice(pl.len_sum.to_native() as usize);
        let mut positions: Box<[MaybeUninit<u16>]> =
            Box::new_uninit_slice(pl.len_sum.to_native() as usize);

        let doc_ids = pl.doc_ids.as_slice();
        let roaringish = pl.positions.as_slice();
        unsafe { assume(doc_ids.len() == roaringish.len()) }

        let mut b = 0;
        for (doc_id, roaringish) in doc_ids.iter().zip(roaringish.iter()) {
            let doc_id = make_doc_id(doc_id.to_native());

            let doc_id_groups =
                unsafe { doc_id_groups.get_unchecked_mut(b..(b + roaringish.inner.len())) };
            let positions = unsafe { positions.get_unchecked_mut(b..(b + roaringish.inner.len())) };
            b += roaringish.inner.len();

            for ((doc_id_group, position), packed) in doc_id_groups
                .iter_mut()
                .zip(positions.iter_mut())
                .zip(roaringish.inner.iter())
            {
                let group = get_group(packed.to_native()) as u64;
                let encoded_positions = get_encoded_positions(packed.to_native()) as u16;
                doc_id_group.write(doc_id | group);
                position.write(encoded_positions);
            }
        }

        let doc_id_groups = unsafe { doc_id_groups.assume_init() };
        let positions = unsafe { positions.assume_init() };
        Self {
            doc_id_groups,
            positions,
        }
    }
}

impl Binary for RoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.positions.iter()) {
            list.entry_with(|f| {
                let doc_id = get_doc_id(*doc_id_group);
                let group = get_group_from_doc_id_group(*doc_id_group);
                f.write_fmt(format_args!(
                    "{doc_id:032b} {group:016b} {encoded_values:016b}"
                ))
            });
        }

        list.finish()
    }
}

// impl Debug for RoaringishPacked {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut list = f.debug_list();
//         for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.positions.iter()) {
//             list.entry_with(|f| {
//                 let doc_id = get_doc_id(*doc_id_group);
//                 let group = get_group_from_doc_id_group(*doc_id_group);
//                 f.debug_tuple("")
//                     .field(&doc_id)
//                     .field(&group)
//                     .field_with(|f| {
//                         f.debug_list()
//                             .entries(
//                                 (0..16u32)
//                                     .filter_map(|i| ((encoded_values >> i) & 1 == 1).then_some(i)),
//                             )
//                             .finish()
//                     })
//                     .finish()
//             });
//         }
//         list.finish()
//     }
// }

impl Display for RoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let it = self
            .doc_id_groups
            .iter()
            .zip(self.positions.iter())
            .map(|(doc_id_group, encoded_values)| {
                let doc_id = get_doc_id(*doc_id_group);
                let group = get_group_from_doc_id_group(*doc_id_group);
                let s = group * 16;
                (0..16u32)
                    .filter_map(move |i| ((encoded_values >> i) & 1 == 1).then_some(i))
                    .map(move |i| (doc_id, s + i))
            })
            .flatten();
        f.debug_list().entries(it).finish()
    }
}

impl<'a> Arbitrary<'a> for RoaringishPacked {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.arbitrary_len::<(u64, u16)>()?;
        let doc_id_groups: Result<Vec<u64>, _> = u.arbitrary_iter()?.take(len).collect();
        let mut doc_id_groups = doc_id_groups?;
        doc_id_groups.sort_unstable();
        doc_id_groups.dedup();

        match doc_id_groups.last().copied() {
            Some(v) => {
                if v == u64::MAX {
                    doc_id_groups.pop();
                }
            },
            None => {},
        }

        let positions: Result<Vec<u16>, _> =
            u.arbitrary_iter()?.take(doc_id_groups.len()).collect();
        let positions = positions?;

        if doc_id_groups.len() != positions.len() {
            return Err(arbitrary::Error::NotEnoughData);
        }

        Ok(Self {
            doc_id_groups: doc_id_groups.into_boxed_slice(),
            positions: positions.into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roaringish() {
        let pos = vec![1, 5, 20, 21, 31, 100, 340];
        let r = Roaringish::from_positions_sorted(pos);
        println!("{r:b}");
        println!("{r:?}");
        println!("{r}");
    }

    #[test]
    fn roaringish_packed() {
        let doc_ids = vec![0, 1, 10];
        let positions = vec![
            Roaringish::from_positions_sorted(vec![1, 5, 20]),
            Roaringish::from_positions_sorted(vec![300]),
            Roaringish::from_positions_sorted(vec![10, 20]),
        ];
        println!("{}", positions[0]);
        println!("{}", positions[1]);
        println!("{}", positions[2]);
        let pl = PostingList::new(doc_ids, positions);
        let r = RoaringishPacked::from(&pl);
        println!("{:#b}", r);
    }

    // #[test]
    // fn roaringish_packed_intersect() {
    //     let doc_ids0 = vec![0, 1, 2, 10, 12];
    //     let positions0 = vec![
    //         Roaringish::from_positions_sorted(vec![1, 5, 20]),
    //         Roaringish::from_positions_sorted(vec![300]),
    //         Roaringish::from_positions_sorted(vec![1]),
    //         Roaringish::from_positions_sorted(vec![10, 20, 31, 34, 49, 63]),
    //         Roaringish::from_positions_sorted(vec![15]),
    //     ];
    //     let pl0 = PostingList::new(doc_ids0, positions0);
    //     let r0 = RoaringishPacked::from(&pl0);

    //     let doc_ids1 = vec![0, 1, 2, 10, 12];
    //     let positions1 = vec![
    //         Roaringish::from_positions_sorted(vec![2, 6]),
    //         Roaringish::from_positions_sorted(vec![301]),
    //         Roaringish::from_positions_sorted(vec![3]),
    //         Roaringish::from_positions_sorted(vec![11, 21, 32, 35, 50, 64]),
    //         Roaringish::from_positions_sorted(vec![16]),
    //     ];
    //     let pl1 = PostingList::new(doc_ids1, positions1);
    //     let r1 = RoaringishPacked::from(&pl1);

    //     println!("{r0:#b}");
    //     println!("{r1:#b}");

    //     let intersection = r0.intersect(&r1, 1);
    //     let doc_ids = intersection.get_doc_ids();
    //     println!("{intersection}");
    //     println!("{doc_ids:?}");
    // }

    #[test]
    fn add_roaringish_packed() {
        let doc_ids0 = vec![0];
        let positions0 = vec![Roaringish::from_positions_sorted(vec![15, 14])];
        let pl0 = PostingList::new(doc_ids0, positions0);
        let r0 = RoaringishPacked::from(&pl0);
        let r1 = &r0 + 1;
        println!("{r0:b}");
        println!("{r1:b}");
    }
}
