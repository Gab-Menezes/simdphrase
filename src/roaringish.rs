pub mod intersect;

use arbitrary::{Arbitrary, Unstructured};
use rkyv::{with::InlineAsBox, Archive, Serialize};
use std::{
    arch::x86_64::_mm256_mask_compressstoreu_epi32,
    fmt::{Binary, Debug, Display},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZero,
    ops::{Add, Deref},
    simd::{
        cmp::SimdPartialEq,
        num::{SimdInt, SimdUint},
        LaneCount, Simd, SupportedLaneCount,
    },
    sync::atomic::Ordering::Relaxed,
};

use crate::Stats;
use crate::{allocator::Aligned64, binary_search::BinarySearchIntersect};

use self::intersect::Intersect;

pub const MAX_VALUE: u32 = 16u32 * u16::MAX as u32;
pub const ADD_ONE_GROUP: u64 = u16::MAX as u64 + 1;

const fn group(val: u32) -> u16 {
    (val / 16) as u16
}

const fn value(val: u32) -> u16 {
    (val % 16) as u16
}

const fn gv(val: u32) -> (u16, u16) {
    (group(val), value(val))
}

const fn pack_doc_id(doc_id: u32) -> u64 {
    (doc_id as u64) << 32
}

const fn pack_group(group: u16) -> u64 {
    (group as u64) << 16
}

const fn pack_value(value: u16) -> u64 {
    1 << value
}

const fn pack_doc_id_group(packed_doc_id: u64, group: u16) -> u64 {
    packed_doc_id | pack_group(group)
}

const fn pack(packed_doc_id: u64, group: u16, value: u16) -> u64 {
    pack_doc_id_group(packed_doc_id, group) | pack_value(value)
}

const fn clear_values(packed: u64) -> u64 {
    packed & !0xFFFF
}

#[inline(always)]
fn clear_values_simd<const N: usize>(packed: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    packed & Simd::splat(!0xFFFF)
}

const fn unpack_doc_id(packed: u64) -> u32 {
    (packed >> 32) as u32
}

#[allow(unused)]
#[inline(always)]
fn unpack_doc_id_simd<const N: usize>(packed: Simd<u64, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    (packed >> Simd::splat(32)).cast()
}

const fn unpack_group(packed: u64) -> u16 {
    (packed >> 16) as u16
}

const fn unpack_values(packed: u64) -> u16 {
    packed as u16
}

#[inline(always)]
fn unpack_values_simd<const N: usize>(packed: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    packed & Simd::splat(0xFFFF)
}

pub enum RoaringishPackedKind<'a, A> {
    Owned(RoaringishPacked),
    Archived(&'a ArchivedBorrowRoaringishPacked<'a, A>),
}

impl<'a, A> RoaringishPackedKind<'a, A> {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            RoaringishPackedKind::Owned(packed) => unsafe {
                let (l, packed, r) = packed.0.align_to::<u8>();
                assert!(l.is_empty());
                assert!(r.is_empty());
                packed
            },
            RoaringishPackedKind::Archived(packed) => unsafe {
                let (l, packed, r) = packed.0.align_to::<u8>();
                assert!(l.is_empty());
                assert!(r.is_empty());
                packed
            },
        }
    }

    pub fn concat<'b: 'a>(self, other: RoaringishPackedKind<'b, A>) -> RoaringishPackedKind<'b, A> {
        unsafe fn copy_data<T, U>(dest: &mut [MaybeUninit<T>], lhs: &[U], rhs: &[U]) {
            let (l, buf, r) = dest.align_to_mut::<MaybeUninit<u8>>();
            assert!(l.is_empty());
            assert!(r.is_empty());

            let (l, lhs, r) = lhs.align_to::<MaybeUninit<u8>>();
            assert!(l.is_empty());
            assert!(r.is_empty());

            let (l, rhs, r) = rhs.align_to::<MaybeUninit<u8>>();
            assert!(l.is_empty());
            assert!(r.is_empty());

            buf[0..lhs.len()].copy_from_slice(lhs);
            buf[lhs.len()..].copy_from_slice(rhs);
        }

        let r = match (self, other) {
            (RoaringishPackedKind::Owned(mut lhs), RoaringishPackedKind::Archived(rhs)) => {
                lhs.0.extend(rhs.0.iter().map(|v| v.to_native()));
                lhs
            }
            (RoaringishPackedKind::Archived(lhs), RoaringishPackedKind::Archived(rhs)) => {
                let n = lhs.0.len() + rhs.0.len();
                let mut packed: Box<[MaybeUninit<u64>], _> =
                    Box::new_uninit_slice_in(n, Aligned64::default());

                unsafe {
                    copy_data(&mut packed, &lhs.0, &rhs.0);
                    let (p_packed, a0) = Box::into_raw_with_allocator(packed);
                    RoaringishPacked(Vec::from_raw_parts_in(p_packed as *mut _, n, n, a0))
                }
            }
            _ => panic!("This type of append should never happen"),
        };
        RoaringishPackedKind::Owned(r)
    }
}

#[derive(PartialEq, Eq, Debug, Serialize, Archive)]
#[repr(transparent)]
pub struct RoaringishPacked(Vec<u64, Aligned64>);

impl Deref for RoaringishPacked {
    type Target = Vec<u64, Aligned64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RoaringishPacked {
    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<u64>()
    }

    pub(crate) fn push(&mut self, doc_id: u32, pos: &[u32]) {
        let packed_doc_id = pack_doc_id(doc_id);

        let mut it = pos.iter().copied();
        let Some(p) = it.next() else {
            return;
        };

        self.0.reserve(pos.len());

        unsafe {
            let (group, value) = gv(p);
            let packed = pack(packed_doc_id, group, value);

            self.0.push_within_capacity(packed).unwrap_unchecked();
        }

        for p in it {
            let (group, value) = gv(p);
            let doc_id_group = pack_doc_id_group(packed_doc_id, group);
            let value = pack_value(value);
            let packed = doc_id_group | value;

            let last_doc_id_group = unsafe { clear_values(*self.0.last().unwrap_unchecked()) };
            if last_doc_id_group == doc_id_group {
                unsafe {
                    *self.0.last_mut().unwrap_unchecked() |= value;
                };
            } else {
                unsafe {
                    self.0.push_within_capacity(packed).unwrap_unchecked();
                }
            }
        }
    }
}

impl Default for RoaringishPacked {
    fn default() -> Self {
        Self(Vec::new_in(Aligned64::default()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aligned;
#[derive(Clone, Copy, Debug)]
pub struct Unaligned;

#[derive(Clone, Copy, Debug, Serialize, Archive)]
pub struct BorrowRoaringishPacked<'a, A>(#[rkyv(with = InlineAsBox)] &'a [u64], PhantomData<A>);

impl<A> Deref for BorrowRoaringishPacked<'_, A> {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> BorrowRoaringishPacked<'a, Aligned> {
    pub fn new_raw(packed: &'a [u64]) -> Self {
        assert!(packed.as_ptr().is_aligned_to(64));
        Self(packed, PhantomData)
    }

    pub fn new(packed: &'a Vec<u64, Aligned64>) -> Self {
        Self(packed, PhantomData)
    }

    #[inline(never)]
    pub fn intersect<I: Intersect>(
        self,
        mut rhs: Self,
        lhs_len: u16,
        stats: &Stats,
    ) -> RoaringishPacked {
        const BINARY_SEARCH_INTERSECT: usize = 650;

        #[inline(always)]
        fn binary_search(
            lhs: &mut BorrowRoaringishPacked<'_, Aligned>,
            rhs: &mut BorrowRoaringishPacked<'_, Aligned>,
        ) {
            // skip the begining of the slice
            let Some(first_lhs) = lhs.0.first() else {
                return;
            };

            let Some(first_rhs) = rhs.0.first() else {
                return;
            };

            let first_lhs = clear_values(*first_lhs);
            let first_rhs = clear_values(*first_rhs);

            if first_lhs < first_rhs {
                let i = match lhs.0.binary_search_by_key(&first_rhs, |p| clear_values(*p)) {
                    // in the case where we are moving the lhs and both values are
                    // equal we need to go back one to ensure that the check of msb
                    // can't skip the previous value
                    Ok(i) => i.saturating_sub(1),
                    Err(i) => i,
                };
                let aligned_i = i / 8 * 8;
                *lhs = BorrowRoaringishPacked::new_raw(&lhs.0[aligned_i..]);
            } else if first_lhs > first_rhs {
                let i = match rhs.0.binary_search_by_key(&first_lhs, |p| clear_values(*p)) {
                    Ok(i) => i,
                    Err(i) => i,
                };
                let aligned_i = i / 8 * 8;
                *rhs = BorrowRoaringishPacked::new_raw(&rhs.0[aligned_i..]);
            }

            // skip the ending of the slice
            // let last_lhs = clear_values(*lhs.0.last().unwrap());
            // let last_rhs = clear_values(*rhs.0.last().unwrap());

            // if last_lhs < last_rhs {
            //     let i = match rhs.0.binary_search_by_key(&last_lhs, |p| clear_values(*p)) {
            //         Ok(i) => i + 1,
            //         Err(i) => i,
            //     } + 1;
            //     *rhs = BorrowRoaringishPacked::new_raw(&rhs.0[..i]);
            // } else if last_lhs > last_rhs {
            //     let i = match lhs.0.binary_search_by_key(&last_rhs, |p| clear_values(*p)) {
            //         Ok(i) => i + 1,
            //         Err(i) => i,
            //     } + 1;
            //     *lhs = BorrowRoaringishPacked::new_raw(&lhs.0[..i]);
            // }
        }

        let mut lhs = self;

        if lhs.0.is_empty() || rhs.0.is_empty() {
            return RoaringishPacked::default();
        }

        let b = std::time::Instant::now();
        binary_search(&mut lhs, &mut rhs);
        stats
            .first_binary_search
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let proportion = lhs.len().max(rhs.len()) / lhs.len().min(rhs.len());
        if proportion >= BINARY_SEARCH_INTERSECT {
            let (packed, _) = BinarySearchIntersect::intersect::<true>(lhs, rhs, lhs_len, stats);
            stats
                .first_intersect
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

            return RoaringishPacked(packed);
        }
        let (packed, msb_packed) = I::intersect::<true>(lhs, rhs, lhs_len, stats);
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if !I::needs_second_pass() {
            return RoaringishPacked(packed);
        }

        let mut msb_packed = BorrowRoaringishPacked::new(&msb_packed);

        let b = std::time::Instant::now();
        binary_search(&mut msb_packed, &mut rhs);
        stats
            .second_binary_search
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let (msb_packed, _) = I::intersect::<false>(msb_packed, rhs, lhs_len, stats);
        stats
            .second_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let capacity = packed.len() + msb_packed.len();
        let mut r_packed = Box::new_uninit_slice_in(capacity, Aligned64::default());
        let mut r_i = 0;
        let mut j = 0;
        for pack in packed.iter().copied() {
            unsafe {
                let doc_id_group = clear_values(pack);
                let values = unpack_values(pack);

                while j < msb_packed.len() {
                    let msb_pack = *msb_packed.get_unchecked(j);
                    let msb_doc_id_group = clear_values(msb_pack);
                    let msb_values = unpack_values(msb_pack);
                    j += 1;

                    if msb_doc_id_group >= doc_id_group {
                        j -= 1;
                        break;
                    }

                    if msb_values > 0 {
                        r_packed.get_unchecked_mut(r_i).write(msb_pack);
                        r_i += 1;
                    }
                }

                let write = values > 0;
                if write {
                    r_packed.get_unchecked_mut(r_i).write(pack);
                    r_i += 1;
                }

                {
                    if j >= msb_packed.len() {
                        continue;
                    }

                    let msb_pack = *msb_packed.get_unchecked(j);
                    let msb_doc_id_group = clear_values(msb_pack);
                    let msb_values = unpack_values(msb_pack);
                    j += 1;
                    if msb_doc_id_group != doc_id_group {
                        j -= 1;
                        continue;
                    }

                    if write {
                        // in this case no bit was set in the intersection,
                        // so we can just `or` the new value with the previous one
                        let r = r_packed.get_unchecked_mut(r_i - 1).assume_init_mut();
                        *r |= msb_values as u64;
                        // *r |= msb_pack;
                    } else if msb_values > 0 {
                        r_packed.get_unchecked_mut(r_i).write(msb_pack);
                        r_i += 1;
                    }
                }
            }
        }
        stats
            .first_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        for msb_pack in msb_packed.iter().skip(j).copied() {
            unsafe {
                let msb_values = unpack_values(msb_pack);
                if msb_values > 0 {
                    r_packed.get_unchecked_mut(r_i).write(msb_pack);
                    r_i += 1;
                }
            }
        }
        stats
            .second_result
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        unsafe {
            let (p_packed, a0) = Box::into_raw_with_allocator(r_packed);
            let packed = Vec::from_raw_parts_in(p_packed as *mut _, r_i, capacity, a0);
            RoaringishPacked(packed)
        }
    }
}

impl<A> BorrowRoaringishPacked<'_, A> {
    #[cfg(not(target_feature = "avx512f"))]
    #[inline(always)]
    pub fn get_doc_ids(&self, stats: &Stats) -> Vec<u32> {
        if self.0.is_empty() {
            return Vec::new();
        }

        if self.0.len() == 1 {
            return vec![unpack_doc_id(self.0[0])];
        }

        let b = std::time::Instant::now();

        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(self.0.len());
        let mut i = 0;

        for [packed0, packed1] in self.0.array_windows::<2>() {
            let doc_id0 = unpack_doc_id(*packed0);
            let doc_id1 = unpack_doc_id(*packed1);
            if doc_id0 != doc_id1 {
                unsafe { doc_ids.get_unchecked_mut(i).write(doc_id0) };
                i += 1;
            }
        }

        unsafe {
            doc_ids
                .get_unchecked_mut(i)
                .write(unpack_doc_id(*self.0.last().unwrap_unchecked()))
        };
        i += 1;

        stats
            .get_doc_ids
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        unsafe { Vec::from_raw_parts(Box::into_raw(doc_ids) as *mut _, i, self.0.len()) }
    }

    #[cfg(target_feature = "avx512f")]
    #[inline(always)]
    pub fn get_doc_ids(&self, stats: &Stats) -> Vec<u32> {
        if self.0.is_empty() {
            return Vec::new();
        }

        if self.0.len() == 1 {
            return vec![unpack_doc_id(self.0[0])];
        }

        let b = std::time::Instant::now();

        let mut doc_ids: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(self.0.len());
        let mut i = 0;

        unsafe { doc_ids.get_unchecked_mut(i).write(unpack_doc_id(self.0[0])) };
        i += 1;

        let mut last_doc_id = unpack_doc_id(self.0[0]);
        let (l, m, r) = self.0.as_simd::<8>();
        assert!(l.is_empty());
        for packed in m {
            let doc_id = unpack_doc_id_simd(*packed);
            let rot = doc_id.rotate_elements_right::<1>();
            let first = doc_id.as_array()[0];
            let last = doc_id.as_array()[7];

            let include_first = (first != last_doc_id) as u8;
            let mask = (doc_id.simd_ne(rot).to_bitmask() as u8 & !1) | include_first;

            unsafe {
                // TODO: avoid compressstore on zen4
                _mm256_mask_compressstoreu_epi32(
                    doc_ids.as_mut_ptr().add(i) as *mut _,
                    mask,
                    doc_id.into(),
                );
            }
            i += mask.count_ones() as usize;
            last_doc_id = last;
        }

        let j = r
            .iter()
            .take_while(|packed| unpack_doc_id(**packed) == last_doc_id)
            .count();
        let r = &r[j..];
        for [packed0, packed1] in r.array_windows::<2>() {
            let doc_id0 = unpack_doc_id(*packed0);
            let doc_id1 = unpack_doc_id(*packed1);
            if doc_id0 != doc_id1 {
                unsafe { doc_ids.get_unchecked_mut(i).write(doc_id0) };
                i += 1;
            }
        }

        if !r.is_empty() {
            unsafe {
                doc_ids
                    .get_unchecked_mut(i)
                    .write(unpack_doc_id(*r.last().unwrap_unchecked()))
            };
            i += 1;
        }

        stats
            .get_doc_ids
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        unsafe { Vec::from_raw_parts(Box::into_raw(doc_ids) as *mut _, i, self.0.len()) }
    }
}

impl Binary for RoaringishPacked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for packed in self.0.iter() {
            list.entry_with(|f| {
                let doc_id = unpack_doc_id(*packed);
                let group = unpack_group(*packed);
                let values = unpack_values(*packed);
                f.write_fmt(format_args!("{doc_id:032b} {group:016b} {values:016b}"))
            });
        }

        list.finish()
    }
}

impl<A> Binary for BorrowRoaringishPacked<'_, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for packed in self.0.iter() {
            list.entry_with(|f| {
                let doc_id = unpack_doc_id(*packed);
                let group = unpack_group(*packed);
                let values = unpack_values(*packed);
                f.write_fmt(format_args!("{doc_id:032b} {group:016b} {values:016b}"))
            });
        }

        list.finish()
    }
}

// impl Debug for AlignedRoaringishPacked {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut list = f.debug_list();
//         for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
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

// impl<'a> Debug for AlignedBorrowRoaringishPacked<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut list = f.debug_list();
//         for (doc_id_group, encoded_values) in self.doc_id_groups.iter().zip(self.values.iter()) {
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
        let it = self.0.iter().flat_map(|packed| {
            let doc_id = unpack_doc_id(*packed);
            let group = unpack_group(*packed) as u32;
            let values = unpack_values(*packed);
            let s = group * 16;
            (0..16u32)
                .filter_map(move |i| ((values >> i) & 1 == 1).then_some(i))
                .map(move |i| (doc_id, s + i))
        });
        f.debug_list().entries(it).finish()
    }
}

impl<A> Display for BorrowRoaringishPacked<'_, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let it = self.0.iter().flat_map(|packed| {
            let doc_id = unpack_doc_id(*packed);
            let group = unpack_group(*packed) as u32;
            let values = unpack_values(*packed);
            let s = group * 16;
            (0..16u32)
                .filter_map(move |i| ((values >> i) & 1 == 1).then_some(i))
                .map(move |i| (doc_id, s + i))
        });
        f.debug_list().entries(it).finish()
    }
}

impl<'a> Arbitrary<'a> for RoaringishPacked {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.arbitrary_len::<(u32, NonZero<u16>, NonZero<u16>)>()?;

        let it = u
            .arbitrary_iter::<(u32, NonZero<u16>, NonZero<u16>)>()?
            .take(len);
        let mut packed = Vec::with_capacity_in(len, Aligned64::default());
        for v in it {
            let (doc_id, group, value) = v?;
            let pack = ((doc_id as u64) << 32)
                | (((group.get() ^ u16::MAX) as u64) << 16)
                | value.get() as u64;
            packed.push(pack);
        }
        packed.sort_unstable_by_key(|p| clear_values(*p));
        packed.dedup_by_key(|p| clear_values(*p));

        Ok(Self(packed))
    }
}
