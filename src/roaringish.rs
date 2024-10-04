pub mod intersect;

use arbitrary::{Arbitrary, Unstructured};
use rkyv::{Archive, Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fmt::{Binary, Debug, Display},
    intrinsics::assume,
    mem::MaybeUninit,
    ops::Add,
    sync::atomic::Ordering::Relaxed,
};

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

#[derive(Default, Debug, Serialize, Archive, PartialEq, Eq)]
pub struct RoaringishPacked {
    doc_id_groups: Vec<u64>,
    values: Vec<u16>,
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
            self.values
                .push_within_capacity(value)
                .unwrap_unchecked();
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
                    self.values
                        .push_within_capacity(value)
                        .unwrap_unchecked();
                }
            }
        }
    }
}

pub struct BorrowRoaringishPacked<'a, P: Packed> {
    pub(crate) doc_id_groups: &'a [P::DocIdGroup],
    pub(crate) values: &'a [P::Values],
    _marker: PhantomData<P>,
}

impl<'a, P: Packed> BorrowRoaringishPacked<'a, P> {
    pub fn new(r: &'a P) -> Self {
        Self {
            doc_id_groups: r.doc_id_groups(),
            values: r.values(),
            _marker: PhantomData,
        }
    }

    pub fn from_raw(doc_id_groups: &'a [P::DocIdGroup], values: &'a [P::Values]) -> Self {
        Self {
            doc_id_groups,
            values,
            _marker: PhantomData,
        }
    }
}

impl ArchivedRoaringishPacked {
    pub fn get_doc_ids(&self) -> Vec<u32> {
        if self.doc_id_groups.len() == 0 {
            return Vec::new();
        }

        if self.doc_id_groups.len() == 1 {
            return vec![get_doc_id(self.doc_id_groups[0].to_native())];
        }

        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(self.doc_id_groups.len());
        let mut i = 0;

        for [packed0, packed1] in self.doc_id_groups.array_windows::<2>() {
            let doc_id0 = get_doc_id(packed0.to_native());
            let doc_id1 = get_doc_id(packed1.to_native());
            if doc_id0 != doc_id1 {
                unsafe { doc_ids.get_unchecked_mut(i).write(doc_id0) };
                i += 1;
            }
        }

        unsafe {
            doc_ids.get_unchecked_mut(i).write(get_doc_id(
                self.doc_id_groups.last().unwrap_unchecked().to_native(),
            ))
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
}

impl Add<u32> for &ArchivedRoaringishPacked {
    type Output = RoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        unsafe {
            assume(self.doc_id_groups.len() == self.values.len());
        }

        let n = self.doc_id_groups.len() * 2;
        let mut doc_id_groups = Box::new_uninit_slice(n);
        let mut values = Box::new_uninit_slice(n);
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;
        for (doc_id_group, packed_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
            let new_values = packed_values.to_native().rotate_left(rhs);
            let postions_current_group = new_values & mask;
            let postions_new_group = new_values & bits_to_check;
            if postions_current_group > 0 {
                unsafe {
                    doc_id_groups
                        .get_unchecked_mut(i)
                        .write(doc_id_group.to_native());
                    values.get_unchecked_mut(i).write(postions_current_group);
                    i += 1;
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    doc_id_groups
                        .get_unchecked_mut(i)
                        .write(doc_id_group.to_native() + 1);
                    values.get_unchecked_mut(i).write(postions_new_group);
                    i += 1;
                }
            }
        }

        unsafe {
            RoaringishPacked {
                doc_id_groups: Vec::from_raw_parts(Box::into_raw(doc_id_groups) as *mut _, i, n),
                values: Vec::from_raw_parts(Box::into_raw(values) as *mut _, i, n),
            }
        }
    }
}

impl Add<u32> for &RoaringishPacked {
    type Output = RoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        unsafe {
            assume(self.doc_id_groups.len() == self.values.len());
        }

        let n = self.doc_id_groups.len() * 2;
        let mut doc_id_groups = Box::new_uninit_slice(n);
        let mut values = Box::new_uninit_slice(n);
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;
        for (doc_id_group, packed_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
            let new_values = packed_values.rotate_left(rhs);
            let postions_current_group = new_values & mask;
            let postions_new_group = new_values & bits_to_check;
            if postions_current_group > 0 {
                unsafe {
                    doc_id_groups.get_unchecked_mut(i).write(*doc_id_group);
                    values.get_unchecked_mut(i).write(postions_current_group);
                    i += 1;
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    doc_id_groups.get_unchecked_mut(i).write(*doc_id_group + 1);
                    values.get_unchecked_mut(i).write(postions_new_group);
                    i += 1;
                }
            }
        }

        unsafe {
            RoaringishPacked {
                doc_id_groups: Vec::from_raw_parts(Box::into_raw(doc_id_groups) as *mut _, i, n),
                values: Vec::from_raw_parts(Box::into_raw(values) as *mut _, i, n),
            }
        }
    }
}

pub trait Packed {
    type DocIdGroup: Copy + Into<u64>;
    type Values: Copy + Into<u16>;

    fn doc_id_groups(&self) -> &[Self::DocIdGroup];
    fn values(&self) -> &[Self::Values];

    fn intersect<I: Intersect, R: Packed>(
        &self,
        rhs: &R,
        rhs_len: u32,
        stats: &Stats,
    ) -> RoaringishPacked
    where
        Self: Packed + Sized,
    {
        let lhs = BorrowRoaringishPacked::new(self);
        let rhs = BorrowRoaringishPacked::new(rhs);
        let b = std::time::Instant::now();
        let (doc_id_groups, values_intersect, msb_doc_id_groups) =
            I::intersect::<true, _, R>(&lhs, &rhs);
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let msb_packed =
            BorrowRoaringishPacked::<RoaringishPacked>::from_raw(&msb_doc_id_groups, &[]);
        let b = std::time::Instant::now();
        let (msb_doc_id_groups, msb_values_intersect, _) =
            I::intersect::<false, RoaringishPacked, R>(&msb_packed, &rhs);
        stats
            .second_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let capacity = values_intersect.len() + msb_values_intersect.len();
        let mut r_doc_id_groups = Box::new_uninit_slice(capacity);
        let mut r_values = Box::new_uninit_slice(capacity);
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
            let doc_id_groups =
                Vec::from_raw_parts(Box::into_raw(r_doc_id_groups) as *mut _, r_i, capacity);
            let values =
                Vec::from_raw_parts(Box::into_raw(r_values) as *mut _, r_i, capacity);
            RoaringishPacked {
                doc_id_groups,
                values,
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

impl Packed for RoaringishPacked {
    type DocIdGroup = u64;
    type Values = u16;

    #[inline(always)]
    fn doc_id_groups(&self) -> &[Self::DocIdGroup] {
        &self.doc_id_groups
    }

    #[inline(always)]
    fn values(&self) -> &[Self::Values] {
        &self.values
    }
}

impl Packed for ArchivedRoaringishPacked {
    type DocIdGroup = rkyv::rend::unaligned::u64_ule;
    type Values = rkyv::rend::unaligned::u16_ule;

    #[inline(always)]
    fn doc_id_groups(&self) -> &[Self::DocIdGroup] {
        &self.doc_id_groups
    }

    #[inline(always)]
    fn values(&self) -> &[Self::Values] {
        &self.values
    }
}

impl Binary for RoaringishPacked {
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
            .zip(self.values.iter())
            .flat_map(|(doc_id_group, encoded_values)| {
                let doc_id = get_doc_id(*doc_id_group);
                let group = get_group_from_doc_id_group(*doc_id_group);
                let s = group * 16;
                (0..16u32)
                    .filter_map(move |i| ((encoded_values >> i) & 1 == 1).then_some(i))
                    .map(move |i| (doc_id, s + i))
            });
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

        if let Some(v) = doc_id_groups.last().copied() {
            if v == u64::MAX {
                doc_id_groups.pop();
            }
        }

        let values: Result<Vec<u16>, _> =
            u.arbitrary_iter()?.take(doc_id_groups.len()).collect();
        let values = values?;

        if doc_id_groups.len() != values.len() {
            return Err(arbitrary::Error::NotEnoughData);
        }

        Ok(Self {
            doc_id_groups,
            values,
        })
    }
}
