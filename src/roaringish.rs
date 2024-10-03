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
pub(crate) const MASK_VALUES: u64 = 0xFFFF;
pub(crate) const MASK_GROUP: u64 = 0xFFFF_0000;
pub(crate) const MASK_DOC_ID: u64 = 0xFFFFFFFF_00000000;
pub(crate) const MASK_DOC_ID_GROUP: u64 = MASK_DOC_ID | MASK_GROUP;
pub(crate) const ADD_ONE_GROUP: u64 = 0x0001_0000;

const fn group(val: u32) -> u32 {
    val / 16
}

const fn value(val: u32) -> u32 {
    val % 16
}

const fn make_doc_id(doc_id: u32) -> u64 {
    (doc_id as u64) << 32
}

const fn make_group(pos: u32) -> u64 {
    (group(pos) << 16) as u64
}

const fn make_value(pos: u32) -> u64 {
    (1 << value(pos)) as u64
}

const fn get_doc_id(packed: u64) -> u32 {
    (packed >> 32) as u32
}

const fn get_group(packed: u64) -> u16 {
    ((packed & MASK_GROUP) >> 16) as u16
}

const fn get_values(packed: u64) -> u16 {
    packed as u16
}

#[derive(Default, Debug, Serialize, Archive, PartialEq, Eq)]
pub struct RoaringishPacked {
    packed: Vec<u64>,
}

impl RoaringishPacked {
    pub(crate) fn push(&mut self, doc_id: u32, pos: &[u32]) {
        let doc_id = make_doc_id(doc_id);

        let mut it = pos.iter().copied();
        let Some(p) = it.next() else {
            return;
        };

        self.packed.reserve(pos.len());

        unsafe {
            self.packed
                .push_within_capacity(doc_id | make_group(p) | make_value(p))
                .unwrap_unchecked();
        }

        for p in it {
            let group = make_group(p);
            let value = make_value(p);
            let last = unsafe { self.packed.last_mut().unwrap_unchecked() };
            if group == *last & MASK_GROUP {
                *last |= value;
            } else {
                unsafe {
                    self.packed
                        .push_within_capacity(doc_id | group | value)
                        .unwrap_unchecked();
                }
            }
        }
    }
}


pub struct BorrowRoaringishPacked<'a, P: Packed> {
    pub(crate) packed: &'a [P::Packed],
    _marker: PhantomData<P>,
}

impl<'a, P: Packed> BorrowRoaringishPacked<'a, P> {
    pub fn new(r: &'a P) -> Self {
        Self {
            packed: r.packed(),
            _marker: PhantomData,
        }
    }

    pub fn from_raw(packed: &'a [P::Packed]) -> Self {
        Self {
            packed,
            _marker: PhantomData,
        }
    }
}

impl ArchivedRoaringishPacked {
    pub fn get_doc_ids(&self) -> Vec<u32> {
        // TODO: SIMD this
        if self.packed.len() == 0 {
            return Vec::new();
        }

        if self.packed.len() == 1 {
            return vec![get_doc_id(self.packed[0].to_native())];
        }

        let n = self.packed.len();
        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(n);
        let mut i = 0;

        for [packed0, packed1] in self.packed.array_windows::<2>() {
            let doc_id0 = get_doc_id(packed0.to_native());
            let doc_id1 = get_doc_id(packed1.to_native());
            if doc_id0 != doc_id1 {
                unsafe { doc_ids.get_unchecked_mut(i).write(doc_id0) };
                i += 1;
            }
        }

        unsafe {
            doc_ids.get_unchecked_mut(i).write(get_doc_id(
                self.packed.last().unwrap_unchecked().to_native(),
            ))
        };
        i += 1;

        unsafe { Vec::from_raw_parts(Box::into_raw(doc_ids) as *mut _, i, n) }
    }
}

impl RoaringishPacked {
    pub fn get_doc_ids(&self) -> Vec<u32> {
        // TODO: SIMD this
        if self.packed.len() == 0 {
            return Vec::new();
        }

        if self.packed.len() == 1 {
            return vec![get_doc_id(self.packed[0])];
        }

        let n = self.packed.len();
        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(n);
        let mut i = 0;

        for [packed0, packed1] in self.packed.array_windows::<2>() {
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
                .write(get_doc_id(*self.packed.last().unwrap_unchecked()))
        };
        i += 1;

        unsafe { Vec::from_raw_parts(Box::into_raw(doc_ids) as *mut _, i, n) }
    }
}

impl Add<u32> for &ArchivedRoaringishPacked {
    type Output = RoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        // TODO: SIMD this
        let n = self.packed.len() * 2;
        let mut packed = Box::new_uninit_slice(n);
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;
        for p in self.packed.iter() {
            let doc_id_group = *p & MASK_DOC_ID_GROUP;
            let values = get_values(p.to_native());
            let new_positions = values.rotate_left(rhs);
            let postions_current_group = (new_positions & mask) as u64;
            let postions_new_group = (new_positions & bits_to_check) as u64;

            if postions_current_group > 0 {
                unsafe {
                    packed
                        .get_unchecked_mut(i)
                        .write(doc_id_group | postions_current_group);
                    i += 1;
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    packed
                        .get_unchecked_mut(i)
                        .write((doc_id_group + ADD_ONE_GROUP) | postions_new_group);
                    i += 1;
                }
            }
        }

        unsafe {
            RoaringishPacked {
                packed: Vec::from_raw_parts(Box::into_raw(packed) as *mut _, i, n),
            }
        }
    }
}

impl Add<u32> for &RoaringishPacked {
    type Output = RoaringishPacked;

    fn add(self, rhs: u32) -> Self::Output {
        // TODO: SIMD this
        let n = self.packed.len() * 2;
        let mut packed = Box::new_uninit_slice(n);
        let mut i = 0;

        let mask = u16::MAX << rhs;
        let bits_to_check = !mask;
        for p in self.packed.iter() {
            let doc_id_group = *p & MASK_DOC_ID_GROUP;
            let values = get_values(*p);
            let new_positions = values.rotate_left(rhs);
            let postions_current_group = (new_positions & mask) as u64;
            let postions_new_group = (new_positions & bits_to_check) as u64;

            if postions_current_group > 0 {
                unsafe {
                    packed
                        .get_unchecked_mut(i)
                        .write(doc_id_group | postions_current_group);
                    i += 1;
                }
            }
            if postions_new_group > 0 {
                unsafe {
                    packed
                        .get_unchecked_mut(i)
                        .write((doc_id_group + ADD_ONE_GROUP) | postions_new_group);
                    i += 1;
                }
            }
        }

        unsafe {
            RoaringishPacked {
                packed: Vec::from_raw_parts(Box::into_raw(packed) as *mut _, i, n),
            }
        }
    }
}

pub trait Packed {
    type Packed: Copy + Into<u64>;

    fn packed(&self) -> &[Self::Packed];

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
        let (packed, msb_doc_id_groups) = I::intersect::<true, _, R>(&lhs, &rhs);
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let msb_packed = BorrowRoaringishPacked::<RoaringishPacked>::from_raw(&msb_doc_id_groups);
        let b = std::time::Instant::now();
        let (msb_packed, _) = I::intersect::<false, RoaringishPacked, R>(&msb_packed, &rhs);
        stats
            .second_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let capacity = packed.len() + msb_packed.len();
        let mut r_packed = Box::new_uninit_slice(capacity);
        let mut r_i = 0;
        let mut j = 0;
        for i in 0..packed.len() {
            unsafe {
                let packed = *packed.get_unchecked(i);
                let doc_id_group = packed & MASK_DOC_ID_GROUP;
                let intersection = packed & MASK_VALUES;

                while j < msb_packed.len() {
                    let msb_packed = *msb_packed.get_unchecked(j);
                    let msb_doc_id_group = msb_packed & MASK_DOC_ID_GROUP;
                    let msb_intersection = msb_packed & MASK_VALUES;
                    j += 1;

                    if msb_doc_id_group >= doc_id_group {
                        j -= 1;
                        break;
                    }

                    if msb_intersection > 0 {
                        r_packed.get_unchecked_mut(r_i).write(msb_packed);
                        r_i += 1;
                    }
                }

                let write = intersection > 0;
                if write {
                    r_packed.get_unchecked_mut(r_i).write(packed);
                    r_i += 1;
                }

                {
                    if j >= msb_packed.len() {
                        continue;
                    }

                    let msb_packed = *msb_packed.get_unchecked(j);
                    let msb_doc_id_group = msb_packed & MASK_DOC_ID_GROUP;
                    let msb_intersection = msb_packed & MASK_VALUES;
                    j += 1;
                    if msb_doc_id_group != doc_id_group {
                        j -= 1;
                        continue;
                    }

                    if write {
                        // in this case no bit was set in the intersection,
                        // so we can just `or` the new value with the previous one
                        let r = r_packed.get_unchecked_mut(r_i - 1);
                        let last = r.assume_init();
                        r.write(last | msb_intersection);
                    } else if msb_intersection > 0 {
                        r_packed.get_unchecked_mut(r_i).write(msb_packed);
                        r_i += 1;
                    }
                }
            }
        }
        stats
            .first_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        for i in j..msb_packed.len() {
            unsafe {
                let msb_packed = *msb_packed.get_unchecked(i);
                let msb_intersection = msb_packed & MASK_VALUES;
                if msb_intersection > 0 {
                    r_packed.get_unchecked_mut(r_i).write(msb_packed);
                    r_i += 1;
                }
            }
        }
        stats
            .second_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let packed = RoaringishPacked {
            packed: unsafe {
                Vec::from_raw_parts(Box::into_raw(r_packed) as *mut _, r_i, capacity)
            },
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
    type Packed = u64;

    #[inline(always)]
    fn packed(&self) -> &[Self::Packed] {
        &self.packed
    }
}

impl Packed for ArchivedRoaringishPacked {
    type Packed = rkyv::rend::unaligned::u64_ule;

    #[inline(always)]
    fn packed(&self) -> &[Self::Packed] {
        &self.packed
    }
}

impl Binary for RoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for p in self.packed.iter() {
            list.entry_with(|f| {
                let doc_id = get_doc_id(*p);
                let group = get_group(*p);
                let values = get_values(*p);
                f.write_fmt(format_args!("{doc_id:032b} {group:016b} {values:016b}"))
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
        let it = self.packed.iter().flat_map(|p| {
            let doc_id = get_doc_id(*p);
            let group = get_group(*p);
            let values = get_values(*p);
            let s = (group * 16) as u32;
            (0..16u32)
                .filter_map(move |i| ((values >> i) & 1 == 1).then_some(i))
                .map(move |i| (doc_id, s + i))
        });
        f.debug_list().entries(it).finish()
    }
}

impl<'a> Arbitrary<'a> for RoaringishPacked {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        const N: u64 = u64::MAX << 16;

        let len = u.arbitrary_len::<u64>()?;
        let packed: Result<Vec<u64>, _> = u.arbitrary_iter()?.take(len).collect();
        let mut packed = packed?;
        packed.sort_unstable();
        packed.dedup_by_key(|p| *p & MASK_DOC_ID_GROUP);

        if let Some(p) = packed.last().copied() {
            let doc_id_group = p & MASK_DOC_ID_GROUP;
            if doc_id_group == N {
                packed.pop();
            }
        }

        Ok(Self {
            packed,
        })
    }
}
