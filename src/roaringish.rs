pub mod intersect;

use arbitrary::{Arbitrary, Unstructured};
use rkyv::{Archive, Serialize};
use std::{
    fmt::{Binary, Debug, Display},
    intrinsics::assume,
    mem::MaybeUninit,
    ops::Add,
    sync::atomic::Ordering::Relaxed,
};

use crate::allocator::{Aligned64, AlignedAllocator};
use crate::Stats;

use self::intersect::Intersect;

pub const MAX_VALUE: u32 = 16u32 * u16::MAX as u32;

const fn group(val: u32) -> u32 {
    val / 16
}

const fn value(val: u32) -> u16 {
    (val % 16) as u16
}

const fn gv(val: u32) -> (u32, u16) {
    (group(val), value(val))
}

const fn make_value(value: u16) -> u16 {
    1 << value
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

#[derive(Debug, Default, PartialEq, Eq, Serialize, Archive)]
pub struct RoaringishPacked {
    pub(crate) doc_id_groups: Vec<u64>,
    pub(crate) values: Vec<u16>,
}

impl RoaringishPacked {
    pub fn size_bytes(&self) -> usize {
        self.doc_id_groups.len() * std::mem::size_of::<u64>()
            + self.values.len() * std::mem::size_of::<u16>()
    }

    pub(crate) fn push(&mut self, doc_id: u32, pos: &[u32]) {
        let doc_id = make_doc_id(doc_id);

        let mut it = pos.iter().copied();
        let Some(p) = it.next() else {
            return;
        };

        self.doc_id_groups.reserve(pos.len());
        self.values.reserve(pos.len());

        unsafe {
            let (group, value) = gv(p);
            let doc_id_group = doc_id | group as u64;
            let value = make_value(value);

            self.doc_id_groups
                .push_within_capacity(doc_id_group)
                .unwrap_unchecked();
            self.values.push_within_capacity(value).unwrap_unchecked();
        }

        for p in it {
            let (group, value) = gv(p);
            let doc_id_group = doc_id | group as u64;
            let value = make_value(value);

            let last_doc_id_group = unsafe { *self.doc_id_groups.last().unwrap_unchecked() };
            if last_doc_id_group == doc_id_group {
                unsafe {
                    *self.values.last_mut().unwrap_unchecked() |= value;
                };
            } else {
                unsafe {
                    self.doc_id_groups
                        .push_within_capacity(doc_id_group)
                        .unwrap_unchecked();
                    self.values.push_within_capacity(value).unwrap_unchecked();
                }
            }
        }
    }
}

pub enum RoaringishPackedKind<'a> {
    Owned(RoaringishPacked),
    Archived(&'a ArchivedRoaringishPacked),
}

impl<'a> RoaringishPackedKind<'a> {
    pub fn as_bytes(&self) -> (&[u8], &[u8]) {
        match self {
            RoaringishPackedKind::Owned(packed) => unsafe {
                let (l, doc_id_groups, r) = packed.doc_id_groups.align_to::<u8>();
                debug_assert!(l.is_empty());
                debug_assert!(r.is_empty());
                let (l, values, r) = packed.values.align_to::<u8>();
                debug_assert!(l.is_empty());
                debug_assert!(r.is_empty());
                (doc_id_groups, values)
            },
            RoaringishPackedKind::Archived(packed) => unsafe {
                let (l, doc_id_groups, r) = packed.doc_id_groups.align_to::<u8>();
                debug_assert!(l.is_empty());
                debug_assert!(r.is_empty());
                let (l, values, r) = packed.values.align_to::<u8>();
                debug_assert!(l.is_empty());
                debug_assert!(r.is_empty());
                (doc_id_groups, values)
            },
        }
    }

    pub fn concat<'b: 'a>(self, other: RoaringishPackedKind<'b>) -> RoaringishPackedKind<'b> {
        unsafe fn copy_data<T, U>(dest: &mut [MaybeUninit<T>], lhs: &[U], rhs: &[U]) {
            let (l, buf, r) = dest.align_to_mut::<MaybeUninit<u8>>();
            debug_assert!(l.is_empty());
            debug_assert!(r.is_empty());

            let (l, lhs, r) = lhs.align_to::<MaybeUninit<u8>>();
            debug_assert!(l.is_empty());
            debug_assert!(r.is_empty());

            let (l, rhs, r) = rhs.align_to::<MaybeUninit<u8>>();
            debug_assert!(l.is_empty());
            debug_assert!(r.is_empty());

            buf[0..lhs.len()].copy_from_slice(lhs);
            buf[lhs.len()..].copy_from_slice(rhs);
        }

        let r = match (self, other) {
            (RoaringishPackedKind::Owned(mut lhs), RoaringishPackedKind::Archived(rhs)) => {
                lhs.doc_id_groups.extend(rhs.doc_id_groups.iter().map(|v| v.to_native()));
                lhs.values.extend(rhs.values.iter().map(|v| v.to_native()));
                lhs
            },
            (RoaringishPackedKind::Archived(lhs), RoaringishPackedKind::Archived(rhs)) => {
                let n = lhs.doc_id_groups.len() + rhs.doc_id_groups.len();
                let mut doc_id_groups: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(n);
                let mut values: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(n);

                unsafe {
                    copy_data(&mut doc_id_groups, &lhs.doc_id_groups, &rhs.doc_id_groups);
                    copy_data(&mut values, &lhs.values, &rhs.values);
                    RoaringishPacked {
                        doc_id_groups: Vec::from_raw_parts(
                            Box::into_raw(doc_id_groups) as *mut _,
                            n,
                            n,
                        ),
                        values: Vec::from_raw_parts(Box::into_raw(values) as *mut _, n, n),
                    }
                }
            }
            _ => panic!("This type of append should never happen"),
        };
        RoaringishPackedKind::Owned(r)
    }
}

pub struct AlignedRoaringishPacked {
    pub(crate) doc_id_groups: Vec<u64, Aligned64>,
    pub(crate) values: Vec<u16, Aligned64>,
}

impl Default for AlignedRoaringishPacked {
    fn default() -> Self {
        Self {
            doc_id_groups: Vec::new_in(Aligned64::default()),
            values: Vec::new_in(Aligned64::default()),
        }
    }
}

pub struct AlignedBorrowRoaringishPacked<'a> {
    pub(crate) doc_id_groups: &'a [u64],
    pub(crate) values: &'a [u16],
}

impl<'a> AlignedBorrowRoaringishPacked<'a> {
    pub fn new(packed: &'a AlignedRoaringishPacked) -> Self {
        AlignedBorrowRoaringishPacked {
            doc_id_groups: &packed.doc_id_groups,
            values: &packed.values,
        }
    }

    pub fn new_raw(doc_id_groups: &'a [u64], values: &'a [u16]) -> Self {
        assert!(doc_id_groups.as_ptr().is_aligned_to(64));
        assert!(values.as_ptr().is_aligned_to(16));
        Self {
            doc_id_groups,
            values,
        }
    }

    pub fn new_doc_id_groups(doc_id_groups: &'a Vec<u64, Aligned64>) -> Self {
        Self {
            doc_id_groups: &doc_id_groups,
            values: &[],
        }
    }

    pub fn get_doc_ids(&self) -> Vec<u32> {
        if self.doc_id_groups.is_empty() {
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

    pub fn intersect<I: Intersect>(
        &self,
        rhs: &Self,
        rhs_len: u32,
        stats: &Stats,
    ) -> AlignedRoaringishPacked {
        let lhs = self;

        let b = std::time::Instant::now();
        let (doc_id_groups, values_intersect, msb_doc_id_groups) = I::intersect::<true>(&lhs, &rhs);
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let msb_packed = AlignedBorrowRoaringishPacked::new_doc_id_groups(&msb_doc_id_groups);
        let b = std::time::Instant::now();
        let (msb_doc_id_groups, msb_values_intersect, _) = I::intersect::<false>(&msb_packed, &rhs);
        stats
            .second_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let capacity = values_intersect.len() + msb_values_intersect.len();
        let mut r_doc_id_groups = Box::new_uninit_slice_in(capacity, Aligned64::default());
        let mut r_values = Box::new_uninit_slice_in(capacity, Aligned64::default());
        let mut r_i = 0;
        let mut j = 0;
        for i in 0..values_intersect.len() {
            unsafe {
                let doc_id_group = *doc_id_groups.get_unchecked(i);
                let intersection = *values_intersect.get_unchecked(i);

                while j < msb_values_intersect.len() {
                    let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(j);
                    let msb_intersection = *msb_values_intersect.get_unchecked(j);
                    j += 1;

                    if msb_doc_id_group >= doc_id_group {
                        j -= 1;
                        break;
                    }

                    if msb_intersection > 0 {
                        r_doc_id_groups
                            .get_unchecked_mut(r_i)
                            .write(msb_doc_id_group);
                        r_values.get_unchecked_mut(r_i).write(msb_intersection);
                        r_i += 1;
                    }
                }

                let write = intersection > 0;
                if write {
                    r_doc_id_groups.get_unchecked_mut(r_i).write(doc_id_group);
                    r_values.get_unchecked_mut(r_i).write(intersection);
                    r_i += 1;
                }

                {
                    if j >= msb_values_intersect.len() {
                        continue;
                    }

                    let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(j);
                    let msb_intersection = *msb_values_intersect.get_unchecked(j);
                    j += 1;
                    if msb_doc_id_group != doc_id_group {
                        j -= 1;
                        continue;
                    }

                    if write {
                        // in this case no bit was set in the intersection,
                        // so we can just `or` the new value with the previous one
                        let r = r_values.get_unchecked_mut(r_i - 1).assume_init();
                        r_values
                            .get_unchecked_mut(r_i - 1)
                            .write(r | msb_intersection);
                    } else if msb_intersection > 0 {
                        r_doc_id_groups
                            .get_unchecked_mut(r_i)
                            .write(msb_doc_id_group);
                        r_values.get_unchecked_mut(r_i).write(msb_intersection);
                        r_i += 1;
                    }
                }
            }
        }
        stats
            .first_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        for i in j..msb_values_intersect.len() {
            unsafe {
                let msb_doc_id_group = *msb_doc_id_groups.get_unchecked(i);
                let msb_intersection = *msb_values_intersect.get_unchecked(i);
                if msb_intersection > 0 {
                    r_doc_id_groups
                        .get_unchecked_mut(r_i)
                        .write(msb_doc_id_group);
                    r_values.get_unchecked_mut(r_i).write(msb_intersection);
                    r_i += 1;
                }
            }
        }
        stats
            .second_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let packed = unsafe {
            let (p_doc_id_groups, a0) = Box::into_raw_with_allocator(r_doc_id_groups);
            let doc_id_groups =
                Vec::from_raw_parts_in(p_doc_id_groups as *mut _, r_i, capacity, a0);

            let (p_values, a1) = Box::into_raw_with_allocator(r_values);
            let values = Vec::from_raw_parts_in(p_values as *mut _, r_i, capacity, a1);
            AlignedRoaringishPacked {
                doc_id_groups,
                values,
            }
        };
        if rhs_len > 1 {
            let b = std::time::Instant::now();
            let borrow = AlignedBorrowRoaringishPacked::new(&packed);
            let r = &borrow + (rhs_len - 1);
            stats
                .add_rhs
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
            r
        } else {
            packed
        }
    }
}

impl<'a> Add<u32> for &AlignedBorrowRoaringishPacked<'a> {
    type Output = AlignedRoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        unsafe {
            assume(self.doc_id_groups.len() == self.values.len());
        }

        let n = self.doc_id_groups.len() * 2;
        let mut doc_id_groups = Box::new_uninit_slice_in(n, Aligned64::default());
        let mut values = Box::new_uninit_slice_in(n, Aligned64::default());
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;


        let mut it = self.doc_id_groups.iter().zip(self.values.iter());
        let Some((doc_id_group, packed_values)) = it.next() else {
            return AlignedRoaringishPacked::default();
        };

        let new_values = packed_values.rotate_left(rhs);
        let postions_current_group = new_values & mask;
        let postions_new_group = new_values & bits_to_check;
        let mut highest_doc_id_group = 0;
        if postions_current_group > 0 {
            unsafe {
                doc_id_groups.get_unchecked_mut(i).write(*doc_id_group);
                values.get_unchecked_mut(i).write(postions_current_group);
                i += 1;
            }
            highest_doc_id_group = *doc_id_group;
        }
        if postions_new_group > 0 {
            unsafe {
                doc_id_groups.get_unchecked_mut(i).write(*doc_id_group + 1);
                values.get_unchecked_mut(i).write(postions_new_group);
                i += 1;
            }
            highest_doc_id_group = *doc_id_group + 1;
        }
        assert!(i > 0);

        for (doc_id_group, packed_values) in it {
            let new_values = packed_values.rotate_left(rhs);
            let postions_current_group = new_values & mask;
            let postions_new_group = new_values & bits_to_check;
            if postions_current_group > 0 {
                if *doc_id_group > highest_doc_id_group {
                    unsafe {
                        doc_id_groups.get_unchecked_mut(i).write(*doc_id_group);
                        values.get_unchecked_mut(i).write(postions_current_group);
                        i += 1;
                        highest_doc_id_group = *doc_id_group;
                    }
                } else {
                    unsafe {
                        let r_values = values.get_unchecked_mut(i - 1).assume_init_mut();
                        *r_values |= postions_current_group;
                    }
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    doc_id_groups.get_unchecked_mut(i).write(*doc_id_group + 1);
                    values.get_unchecked_mut(i).write(postions_new_group);
                    i += 1;
                    highest_doc_id_group = *doc_id_group + 1;
                }
            }
        }

        unsafe {
            AlignedRoaringishPacked {
                doc_id_groups: Vec::from_raw_parts_in(
                    Box::into_raw(doc_id_groups) as *mut _,
                    i,
                    n,
                    Aligned64::default(),
                ),
                values: Vec::from_raw_parts_in(
                    Box::into_raw(values) as *mut _,
                    i,
                    n,
                    Aligned64::default(),
                ),
            }
        }
    }
}

impl Binary for AlignedRoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
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

impl<'a> Binary for AlignedBorrowRoaringishPacked<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
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

// impl Debug for AlignedRoaringishPacked {
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

impl Display for AlignedRoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let it = self.doc_id_groups.iter().zip(self.values.iter()).flat_map(
            |(doc_id_group, encoded_values)| {
                let doc_id = get_doc_id(*doc_id_group);
                let group = get_group_from_doc_id_group(*doc_id_group);
                let s = group * 16;
                (0..16u32)
                    .filter_map(move |i| ((encoded_values >> i) & 1 == 1).then_some(i))
                    .map(move |i| (doc_id, s + i))
            },
        );
        f.debug_list().entries(it).finish()
    }
}

impl<'a> Display for AlignedBorrowRoaringishPacked<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let it = self.doc_id_groups.iter().zip(self.values.iter()).flat_map(
            |(doc_id_group, encoded_values)| {
                let doc_id = get_doc_id(*doc_id_group);
                let group = get_group_from_doc_id_group(*doc_id_group);
                let s = group * 16;
                (0..16u32)
                    .filter_map(move |i| ((encoded_values >> i) & 1 == 1).then_some(i))
                    .map(move |i| (doc_id, s + i))
            },
        );
        f.debug_list().entries(it).finish()
    }
}

impl<'a> Arbitrary<'a> for AlignedRoaringishPacked {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.arbitrary_len::<(u64, u16)>()?;

        let it = u.arbitrary_iter()?.take(len);
        let mut doc_id_groups: Vec<u64, Aligned64> =
            Vec::with_capacity_in(len, Aligned64::default());
        for v in it {
            doc_id_groups.push(v?);
        }
        doc_id_groups.sort_unstable();
        doc_id_groups.dedup();

        if let Some(v) = doc_id_groups.last().copied() {
            if v == u64::MAX {
                doc_id_groups.pop();
            }
        }

        let it = u.arbitrary_iter()?.take(doc_id_groups.len());
        let mut values: Vec<u16, Aligned64> = Vec::with_capacity_in(len, Aligned64::default());
        for v in it {
            values.push(v?);
        }

        if doc_id_groups.len() != values.len() {
            return Err(arbitrary::Error::NotEnoughData);
        }

        Ok(Self {
            doc_id_groups,
            values,
        })
    }
}
