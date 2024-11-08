pub mod intersect;

use arbitrary::{unstructured::ArbitraryIter, Arbitrary, Unstructured};
use rkyv::{with::{Inline, InlineAsBox, Skip}, Archive, Serialize};
use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_mask_compressstoreu_epi16, _mm512_mask_compressstoreu_epi64,
    },
    fmt::{Binary, Debug, Display},
    intrinsics::assume,
    iter::Take,
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZero,
    ops::Add,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdInt, SimdUint},
        simd_swizzle, Simd, SimdElement,
    },
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

pub enum RoaringishPackedKind<'a, A> {
    Owned(RoaringishPacked),
    Archived(&'a ArchivedBorrowRoaringishPacked<'a, A>),
}

impl<'a, A> RoaringishPackedKind<'a, A> {
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

    pub fn concat<'b: 'a>(self, other: RoaringishPackedKind<'b, A>) -> RoaringishPackedKind<'b, A> {
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
            }
            (RoaringishPackedKind::Archived(lhs), RoaringishPackedKind::Archived(rhs)) => {
                let n = lhs.doc_id_groups.len() + rhs.doc_id_groups.len();
                let mut doc_id_groups: Box<[MaybeUninit<u64>], _> =
                    Box::new_uninit_slice_in(n, Aligned64::default());
                let mut values: Box<[MaybeUninit<u16>], _> =
                    Box::new_uninit_slice_in(n, Aligned64::default());

                unsafe {
                    copy_data(&mut doc_id_groups, &lhs.doc_id_groups, &rhs.doc_id_groups);
                    copy_data(&mut values, &lhs.values, &rhs.values);
                    let (p_doc_id_groups, a0) = Box::into_raw_with_allocator(doc_id_groups);
                    let (p_values, a1) = Box::into_raw_with_allocator(values);
                    RoaringishPacked {
                        doc_id_groups: Vec::from_raw_parts_in(p_doc_id_groups as *mut _, n, n, a0),
                        values: Vec::from_raw_parts_in(p_values as *mut _, n, n, a1),
                    }
                }
            }
            _ => panic!("This type of append should never happen"),
        };
        RoaringishPackedKind::Owned(r)
    }
}

#[derive(PartialEq, Eq, Debug, Serialize, Archive)]
pub struct RoaringishPacked {
    pub doc_id_groups: Vec<u64, Aligned64>,
    pub values: Vec<u16, Aligned64>,
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

impl Default for RoaringishPacked {
    fn default() -> Self {
        Self {
            doc_id_groups: Vec::new_in(Aligned64::default()),
            values: Vec::new_in(Aligned64::default()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aligned;
#[derive(Clone, Copy, Debug)]
pub struct Unaligned;

#[derive(Clone, Copy, Debug, Serialize, Archive)]
pub struct BorrowRoaringishPacked<'a, A> {
    #[rkyv(with = InlineAsBox)]
    pub(crate) doc_id_groups: &'a [u64],
    #[rkyv(with = InlineAsBox)]
    pub(crate) values: &'a [u16],
    _marker: PhantomData<A>,
}

impl<'a> BorrowRoaringishPacked<'a, Aligned> {
    pub fn new(packed: &'a RoaringishPacked) -> Self {
        Self {
            doc_id_groups: &packed.doc_id_groups,
            values: &packed.values,
            _marker: PhantomData,
        }
    }

    pub fn new_raw(doc_id_groups: &'a [u64], values: &'a [u16]) -> Self {
        assert!(doc_id_groups.as_ptr().is_aligned_to(64));
        assert!(values.as_ptr().is_aligned_to(16));
        Self {
            doc_id_groups,
            values,
            _marker: PhantomData,
        }
    }

    pub fn new_aligned(doc_id_groups: &'a Vec<u64, Aligned64>, values: &'a Vec<u16, Aligned64>) -> Self {
        Self {
            doc_id_groups: &doc_id_groups,
            values: values,
            _marker: PhantomData,
        }
    }

    pub fn intersect<I: Intersect>(
        self,
        mut rhs: Self,
        lhs_len: u16,
        stats: &Stats,
    ) -> RoaringishPacked {
        let mut lhs = self;

        if lhs.doc_id_groups.is_empty() || rhs.doc_id_groups.is_empty() {
            return RoaringishPacked::default();
        }

        let b = std::time::Instant::now();
        let first_lhs = lhs.doc_id_groups[0];
        let first_rhs = rhs.doc_id_groups[0];

        if first_lhs < first_rhs {
            let i = match rhs.doc_id_groups.binary_search(&first_lhs) {
                Ok(i) => i,
                Err(i) => i,
            };
            let aligned_i = i / 8 * 8;
            rhs = BorrowRoaringishPacked::new_raw(
                &rhs.doc_id_groups[aligned_i..],
                &rhs.values[aligned_i..],
            );
        } else if first_lhs > first_rhs {
            let i = match lhs.doc_id_groups.binary_search(&first_rhs) {
                Ok(i) => i,
                Err(i) => i,
            };
            let aligned_i = i / 8 * 8;
            lhs = BorrowRoaringishPacked::new_raw(
                &lhs.doc_id_groups[aligned_i..],
                &lhs.values[aligned_i..],
            );
        }
        stats
            .binary_search
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let b = std::time::Instant::now();
        let (doc_id_groups, values_intersect, msb_doc_id_groups, msb_values) = I::intersect::<true>(lhs, rhs, lhs_len);
        stats
            .first_intersect
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let msb_packed = BorrowRoaringishPacked::new_aligned(&msb_doc_id_groups, &msb_values);
        let b = std::time::Instant::now();
        let (msb_doc_id_groups, msb_values_intersect, _, _) = I::intersect::<false>(msb_packed, rhs, lhs_len);
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
            RoaringishPacked {
                doc_id_groups,
                values,
            }
        };
        packed
    }
}

impl<'a, A> BorrowRoaringishPacked<'a, A> {
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
}

impl<'a, A> Add<u16> for BorrowRoaringishPacked<'a, A> {
    type Output = RoaringishPacked;

    fn add(self, rhs: u16) -> Self::Output {
        // Right now we only allow values to jump up to 1 group
        assert!(rhs <= 15);
        assert_eq!(self.doc_id_groups.len(), self.values.len());

        let n = self.doc_id_groups.len() * 2;
        let mut doc_id_groups = Box::new_uninit_slice_in(n, Aligned64::default());
        let mut values = Box::new_uninit_slice_in(n, Aligned64::default());
        let mut i = 0;

        let mask_current_group = u16::MAX << rhs;
        let mask_next_group = !mask_current_group;

        let mut it = self.doc_id_groups.iter().zip(self.values.iter());
        let Some((doc_id_group, packed_values)) = it.next() else {
            return RoaringishPacked::default();
        };

        let new_values = packed_values.rotate_left(rhs as u32);
        let postions_current_group = new_values & mask_current_group;
        let postions_new_group = new_values & mask_next_group;
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
            let new_values = packed_values.rotate_left(rhs as u32);
            let postions_current_group = new_values & mask_current_group;
            let postions_new_group = new_values & mask_next_group;
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
            RoaringishPacked {
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

// impl<'a> Add<u16> for &AlignedBorrowRoaringishPacked<'a> {
//     type Output = AlignedRoaringishPacked;

//     fn add(self, rhs: u16) -> Self::Output {
//         const LANES: usize = 8;
//         #[inline(always)]
//         fn rotl_u16(a: &Simd<u16, LANES>, i: u16) -> Simd<u16, LANES> {
//             const N: u16 = 16;
//             let i = i % N;
//             let p0 = a << i;
//             let p1 = a >> (N - i);
//             p0 | p1
//         }

//         #[inline(always)]
//         fn write_first(
//             r_doc_id_groups: &mut Box<[MaybeUninit<u64>], Aligned64>,
//             r_values: &mut Box<[MaybeUninit<u16>], Aligned64>,
//             i: &mut usize,
//             doc_id_group: u64,
//             postions_current_group: u16,
//             postions_new_group: u16,
//         ) {
//             if postions_current_group > 0 {
//                 unsafe {
//                     r_doc_id_groups.get_unchecked_mut(*i).write(doc_id_group);
//                     r_values.get_unchecked_mut(*i).write(postions_current_group);
//                     *i += 1;
//                 }
//             }
//             if postions_new_group > 0 {
//                 unsafe {
//                     r_doc_id_groups
//                         .get_unchecked_mut(*i)
//                         .write(doc_id_group + 1);
//                     r_values.get_unchecked_mut(*i).write(postions_new_group);
//                     *i += 1;
//                 }
//             }
//             assert!(*i > 0);
//         }

//         #[inline(always)]
//         fn write(
//             r_doc_id_groups: &mut Box<[MaybeUninit<u64>], Aligned64>,
//             r_values: &mut Box<[MaybeUninit<u16>], Aligned64>,
//             i: &mut usize,
//             doc_id_group: u64,
//             postions_current_group: u16,
//             postions_new_group: u16,
//         ) {
//             let highest_doc_id_group =
//                 unsafe { r_doc_id_groups.get_unchecked(*i - 1).assume_init() };
//             if postions_current_group > 0 {
//                 if doc_id_group > highest_doc_id_group {
//                     unsafe {
//                         r_doc_id_groups.get_unchecked_mut(*i).write(doc_id_group);
//                         r_values.get_unchecked_mut(*i).write(postions_current_group);
//                         *i += 1;
//                     }
//                 } else {
//                     unsafe {
//                         *r_values.get_unchecked_mut(*i - 1).assume_init_mut() |=
//                             postions_current_group;
//                     }
//                 }
//             }
//             if postions_new_group > 0 {
//                 unsafe {
//                     r_doc_id_groups
//                         .get_unchecked_mut(*i)
//                         .write(doc_id_group + 1);
//                     r_values.get_unchecked_mut(*i).write(postions_new_group);
//                     *i += 1;
//                 }
//             }
//         }

//         #[inline(always)]
//         fn swizzle<T: SimdElement>(a: Simd<T, LANES>, b: Simd<T, LANES>) -> Simd<T, { LANES * 2 }> {
//             simd_swizzle!(
//                 a,
//                 b,
//                 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,]
//             )
//         }

//         #[inline(always)]
//         fn split<T: SimdElement>(a: Simd<T, { LANES * 2 }>) -> (Simd<T, LANES>, Simd<T, LANES>) {
//             let arr = a.to_array();
//             unsafe {
//                 let low = Simd::from_array(arr[..LANES].try_into().unwrap_unchecked());
//                 let high = Simd::from_array(arr[LANES..].try_into().unwrap_unchecked());
//                 (low, high)
//             }
//         }

//         #[inline(always)]
//         fn calc(
//             doc_id_group: &Simd<u64, LANES>,
//             values: &Simd<u16, LANES>,
//             rhs: u16,
//             splat_current_group_mask: &Simd<u16, LANES>,
//             splat_new_group_mask: &Simd<u16, LANES>,
//         ) -> (
//             Simd<u64, LANES>,
//             u8,
//             Simd<u64, LANES>,
//             u8,
//             Simd<u16, { 2 * LANES }>,
//             u16,
//         ) {
//             let doc_id_group_p_1 = doc_id_group + Simd::splat(1);

//             let new_values = rotl_u16(values, rhs);
//             let postions_current_group = new_values & splat_current_group_mask;
//             let postions_new_group = new_values & splat_new_group_mask;

//             let concated_doc_id_group = swizzle(*doc_id_group, doc_id_group_p_1);
//             let concated_positions = swizzle(postions_current_group, postions_new_group);

//             let rotr_concated_doc_id_group = concated_doc_id_group.rotate_elements_right::<1>();
//             let rotr_concated_positions = concated_positions.rotate_elements_right::<1>();

//             let mask = concated_doc_id_group
//                 .simd_eq(rotr_concated_doc_id_group)
//                 .to_int()
//                 .cast::<u16>();
//             let rotl_mask = mask.rotate_elements_left::<1>();
//             let final_positions =
//                 !rotl_mask & (concated_positions | (mask & rotr_concated_positions));
//             // u16 because 2*LANES
//             let mask = final_positions.simd_gt(Simd::splat(0)).to_bitmask() as u16;
//             let mask_0 = mask as u8;
//             let mask_1 = (mask >> 8) as u8;
//             let (final_doc_id_group_0, final_doc_id_group_1) = split(concated_doc_id_group);

//             (
//                 final_doc_id_group_0,
//                 mask_0,
//                 final_doc_id_group_1,
//                 mask_1,
//                 final_positions,
//                 mask,
//             )
//         }

//         // Right now we only allow values to jump up to 1 group
//         assert!(rhs <= 15);

//         unsafe {
//             assume(self.doc_id_groups.len() == self.values.len());
//         }

//         if self.doc_id_groups.is_empty() {
//             return AlignedRoaringishPacked::default();
//         }

//         let n = self.doc_id_groups.len() * 2;
//         let mut r_doc_id_groups = Box::<[u64], _>::new_uninit_slice_in(n, Aligned64::default());
//         let mut r_values = Box::<[u16], _>::new_uninit_slice_in(n, Aligned64::default());
//         let mut i = 0;

//         let current_group_mask = u16::MAX << rhs;
//         let new_group_mask = !current_group_mask;

//         let splat_current_group_mask = Simd::splat(current_group_mask);
//         let splat_new_group_mask = Simd::splat(new_group_mask);

//         let (p_doc_id_groups, doc_id_groups, rem_doc_id_groups) =
//             self.doc_id_groups.as_simd::<LANES>();
//         let (p_values, values, rem_values) = self.values.as_simd::<LANES>();
//         assert!(p_doc_id_groups.is_empty());
//         assert!(p_values.is_empty());
//         assert_eq!(rem_doc_id_groups.len(), rem_values.len());

//         let mut it = doc_id_groups.into_iter().zip(values);

//         let Some((doc_id_group, values)) = it.next() else {
//             let new_values = rem_values[0].rotate_left(rhs as u32);
//             write_first(
//                 &mut r_doc_id_groups,
//                 &mut r_values,
//                 &mut i,
//                 rem_doc_id_groups[0],
//                 new_values & current_group_mask,
//                 new_values & new_group_mask,
//             );

//             for (doc_id_group, values) in rem_doc_id_groups[1..].into_iter().zip(&rem_values[1..]) {
//                 let new_values = values.rotate_left(rhs as u32);
//                 write(
//                     &mut r_doc_id_groups,
//                     &mut r_values,
//                     &mut i,
//                     *doc_id_group,
//                     new_values & current_group_mask,
//                     new_values & new_group_mask,
//                 );
//             }

//             return unsafe {
//                 AlignedRoaringishPacked {
//                     doc_id_groups: Vec::from_raw_parts_in(
//                         Box::into_raw(r_doc_id_groups) as *mut _,
//                         i,
//                         n,
//                         Aligned64::default(),
//                     ),
//                     values: Vec::from_raw_parts_in(
//                         Box::into_raw(r_values) as *mut _,
//                         i,
//                         n,
//                         Aligned64::default(),
//                     ),
//                 }
//             };
//         };

//         let (final_doc_id_group_0, mask_0, final_doc_id_group_1, mask_1, final_positions, mask) =
//             calc(
//                 doc_id_group,
//                 values,
//                 rhs,
//                 &splat_current_group_mask,
//                 &splat_new_group_mask,
//             );

//         unsafe {
//             _mm256_mask_compressstoreu_epi16(
//                 r_values.as_mut_ptr().add(i) as *mut _,
//                 mask,
//                 final_positions.into(),
//             );

//             _mm512_mask_compressstoreu_epi64(
//                 r_doc_id_groups.as_mut_ptr().add(i) as *mut _,
//                 mask_0,
//                 final_doc_id_group_0.into(),
//             );

//             i += mask_0.count_ones() as usize;

//             _mm512_mask_compressstoreu_epi64(
//                 r_doc_id_groups.as_mut_ptr().add(i) as *mut _,
//                 mask_1,
//                 final_doc_id_group_1.into(),
//             );

//             i += mask_1.count_ones() as usize;
//         }

//         for (doc_id_group, values) in it {
//             let last_doc_id_groups =
//                 unsafe { r_doc_id_groups.get_unchecked_mut(i - 1).assume_init() };
//             // is it ok to hold &mut and write through *mut using compress store ?
//             let last_value = unsafe { r_values.get_unchecked_mut(i - 1).assume_init() };
//             let fst = doc_id_group.as_array()[0];

//             let (final_doc_id_group_0, mask_0, final_doc_id_group_1, mask_1, final_positions, mask) =
//                 calc(
//                     doc_id_group,
//                     values,
//                     rhs,
//                     &splat_current_group_mask,
//                     &splat_new_group_mask,
//                 );

//             // TODO: maybe try branchless
//             if fst == last_doc_id_groups && (mask & 1) == 1 {
//                 unsafe {
//                     _mm256_mask_compressstoreu_epi16(
//                         r_values.as_mut_ptr().add(i - 1) as *mut _,
//                         mask,
//                         final_positions.into(),
//                     );
//                     *r_values.get_unchecked_mut(i - 1).assume_init_mut() |= last_value;

//                     _mm512_mask_compressstoreu_epi64(
//                         r_doc_id_groups.as_mut_ptr().add(i - 1) as *mut _,
//                         mask_0,
//                         final_doc_id_group_0.into(),
//                     );

//                     i += mask_0.count_ones() as usize;

//                     _mm512_mask_compressstoreu_epi64(
//                         r_doc_id_groups.as_mut_ptr().add(i - 1) as *mut _,
//                         mask_1,
//                         final_doc_id_group_1.into(),
//                     );

//                     i += mask_1.count_ones() as usize;
//                     i -= 1;
//                 }
//             } else {
//                 unsafe {
//                     _mm256_mask_compressstoreu_epi16(
//                         r_values.as_mut_ptr().add(i) as *mut _,
//                         mask,
//                         final_positions.into(),
//                     );

//                     _mm512_mask_compressstoreu_epi64(
//                         r_doc_id_groups.as_mut_ptr().add(i) as *mut _,
//                         mask_0,
//                         final_doc_id_group_0.into(),
//                     );

//                     i += mask_0.count_ones() as usize;

//                     _mm512_mask_compressstoreu_epi64(
//                         r_doc_id_groups.as_mut_ptr().add(i) as *mut _,
//                         mask_1,
//                         final_doc_id_group_1.into(),
//                     );

//                     i += mask_1.count_ones() as usize;
//                 }
//             }
//         }

//         for (doc_id_group, values) in rem_doc_id_groups.into_iter().zip(rem_values) {
//             let new_values = values.rotate_left(rhs as u32);
//             write(
//                 &mut r_doc_id_groups,
//                 &mut r_values,
//                 &mut i,
//                 *doc_id_group,
//                 new_values & current_group_mask,
//                 new_values & new_group_mask,
//             );
//         }

//         unsafe {
//             AlignedRoaringishPacked {
//                 doc_id_groups: Vec::from_raw_parts_in(
//                     Box::into_raw(r_doc_id_groups) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//                 values: Vec::from_raw_parts_in(
//                     Box::into_raw(r_values) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//             }
//         }
//     }
// }

// impl<'a> Add<u16> for &AlignedBorrowRoaringishPacked<'a> {
//     type Output = AlignedRoaringishPacked;

//     fn add(self, rhs: u16) -> Self::Output {
//         const LANES: usize = 8;
//         #[inline(always)]
//         fn rotl_u16(a: &Simd<u16, LANES>, i: u16) -> Simd<u16, LANES> {
//             const N: u16 = 16;
//             let i = i % N;
//             let p0 = a << i;
//             let p1 = a >> (N - i);
//             p0 | p1
//         }

//         #[inline(always)]
//         fn write_first(
//             r_doc_id_groups: &mut Box<[MaybeUninit<u64>], Aligned64>,
//             r_values: &mut Box<[MaybeUninit<u16>], Aligned64>,
//             i: &mut usize,
//             doc_id_group: u64,
//             postions_current_group: u16,
//             postions_new_group: u16,
//         ) {
//             if postions_current_group > 0 {
//                 unsafe {
//                     r_doc_id_groups.get_unchecked_mut(*i).write(doc_id_group);
//                     r_values.get_unchecked_mut(*i).write(postions_current_group);
//                     *i += 1;
//                 }
//             }
//             if postions_new_group > 0 {
//                 unsafe {
//                     r_doc_id_groups
//                         .get_unchecked_mut(*i)
//                         .write(doc_id_group + 1);
//                     r_values.get_unchecked_mut(*i).write(postions_new_group);
//                     *i += 1;
//                 }
//             }
//             assert!(*i > 0);
//         }

//         #[inline(always)]
//         fn write(
//             r_doc_id_groups: &mut Box<[MaybeUninit<u64>], Aligned64>,
//             r_values: &mut Box<[MaybeUninit<u16>], Aligned64>,
//             i: &mut usize,
//             doc_id_group: u64,
//             postions_current_group: u16,
//             postions_new_group: u16,
//         ) {
//             let highest_doc_id_group =
//                 unsafe { r_doc_id_groups.get_unchecked(*i - 1).assume_init() };
//             if postions_current_group > 0 {
//                 if doc_id_group > highest_doc_id_group {
//                     unsafe {
//                         r_doc_id_groups.get_unchecked_mut(*i).write(doc_id_group);
//                         r_values.get_unchecked_mut(*i).write(postions_current_group);
//                         *i += 1;
//                     }
//                 } else {
//                     unsafe {
//                         *r_values.get_unchecked_mut(*i - 1).assume_init_mut() |=
//                             postions_current_group;
//                     }
//                 }
//             }
//             if postions_new_group > 0 {
//                 unsafe {
//                     r_doc_id_groups
//                         .get_unchecked_mut(*i)
//                         .write(doc_id_group + 1);
//                     r_values.get_unchecked_mut(*i).write(postions_new_group);
//                     *i += 1;
//                 }
//             }
//         }

//         // Right now we only allow values to jump up to 1 group
//         assert!(rhs <= 15);

//         unsafe {
//             assume(self.doc_id_groups.len() == self.values.len());
//         }

//         if self.doc_id_groups.is_empty() {
//             return AlignedRoaringishPacked::default();
//         }

//         let n = self.doc_id_groups.len() * 2;
//         let mut r_doc_id_groups = Box::<[u64], _>::new_uninit_slice_in(n, Aligned64::default());
//         let mut r_values = Box::<[u16], _>::new_uninit_slice_in(n, Aligned64::default());
//         let mut i = 0;

//         let current_group_mask = u16::MAX << rhs;
//         let new_group_mask = !current_group_mask;

//         let splat_current_group_mask = Simd::splat(current_group_mask);
//         let splat_new_group_mask = Simd::splat(new_group_mask);

//         let doc_id_groups = self.doc_id_groups.array_chunks::<LANES>();
//         let rem_doc_id_groups = doc_id_groups.remainder();
//         let (p_values, values, rem_values) = self.values.as_simd::<LANES>();
//         assert!(p_values.is_empty());
//         assert_eq!(rem_doc_id_groups.len(), rem_values.len());

//         let mut it = doc_id_groups.into_iter().zip(values);

//         let Some((doc_id_group, values)) = it.next() else {
//             let new_values = rem_values[0].rotate_left(rhs as u32);
//             write_first(
//                 &mut r_doc_id_groups,
//                 &mut r_values,
//                 &mut i,
//                 rem_doc_id_groups[0],
//                 new_values & current_group_mask,
//                 new_values & new_group_mask,
//             );

//             for (doc_id_group, values) in rem_doc_id_groups[1..].into_iter().zip(&rem_values[1..]) {
//                 let new_values = values.rotate_left(rhs as u32);
//                 write(
//                     &mut r_doc_id_groups,
//                     &mut r_values,
//                     &mut i,
//                     *doc_id_group,
//                     new_values & current_group_mask,
//                     new_values & new_group_mask,
//                 );
//             }

//             return unsafe {
//                 AlignedRoaringishPacked {
//                     doc_id_groups: Vec::from_raw_parts_in(
//                         Box::into_raw(r_doc_id_groups) as *mut _,
//                         i,
//                         n,
//                         Aligned64::default(),
//                     ),
//                     values: Vec::from_raw_parts_in(
//                         Box::into_raw(r_values) as *mut _,
//                         i,
//                         n,
//                         Aligned64::default(),
//                     ),
//                 }
//             };
//         };

//         let new_values = rotl_u16(values, rhs);
//         let postions_current_group = new_values & splat_current_group_mask;
//         let postions_new_group = new_values & splat_new_group_mask;

//         let postions_current_group = postions_current_group.as_array();
//         let postions_new_group = postions_new_group.as_array();

//         write_first(
//             &mut r_doc_id_groups,
//             &mut r_values,
//             &mut i,
//             doc_id_group[0],
//             postions_current_group[0],
//             postions_new_group[0],
//         );

//         for ((doc_id_group, postions_current_group), postions_new_group) in doc_id_group[1..]
//             .into_iter()
//             .zip(&postions_current_group[1..])
//             .zip(&postions_new_group[1..])
//         {
//             write(
//                 &mut r_doc_id_groups,
//                 &mut r_values,
//                 &mut i,
//                 *doc_id_group,
//                 *postions_current_group,
//                 *postions_new_group,
//             );
//         }

//         for (doc_id_group, values) in it {
//             let new_values = rotl_u16(values, rhs);
//             let postions_current_group = new_values & splat_current_group_mask;
//             let postions_new_group = new_values & splat_new_group_mask;

//             let postions_current_group = postions_current_group.as_array();
//             let postions_new_group = postions_new_group.as_array();
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[0],
//             //     postions_current_group[0],
//             //     postions_new_group[0],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[1],
//             //     postions_current_group[1],
//             //     postions_new_group[1],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[2],
//             //     postions_current_group[2],
//             //     postions_new_group[2],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[3],
//             //     postions_current_group[3],
//             //     postions_new_group[3],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[4],
//             //     postions_current_group[4],
//             //     postions_new_group[4],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[5],
//             //     postions_current_group[5],
//             //     postions_new_group[5],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[6],
//             //     postions_current_group[6],
//             //     postions_new_group[6],
//             // );
//             // write(
//             //     &mut r_doc_id_groups,
//             //     &mut r_values,
//             //     &mut i,
//             //     doc_id_group[7],
//             //     postions_current_group[7],
//             //     postions_new_group[7],
//             // );

//             for ((doc_id_group, postions_current_group), postions_new_group) in doc_id_group
//                 .into_iter()
//                 .zip(postions_current_group)
//                 .zip(postions_new_group)
//             {
//                 write(
//                     &mut r_doc_id_groups,
//                     &mut r_values,
//                     &mut i,
//                     *doc_id_group,
//                     *postions_current_group,
//                     *postions_new_group,
//                 );
//             }
//         }

//         for (doc_id_group, values) in rem_doc_id_groups.into_iter().zip(rem_values) {
//             let new_values = values.rotate_left(rhs as u32);
//             write(
//                 &mut r_doc_id_groups,
//                 &mut r_values,
//                 &mut i,
//                 *doc_id_group,
//                 new_values & current_group_mask,
//                 new_values & new_group_mask,
//             );
//         }

//         unsafe {
//             AlignedRoaringishPacked {
//                 doc_id_groups: Vec::from_raw_parts_in(
//                     Box::into_raw(r_doc_id_groups) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//                 values: Vec::from_raw_parts_in(
//                     Box::into_raw(r_values) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//             }
//         }
//     }
// }

// impl<'a> Add<u32> for &AlignedBorrowRoaringishPacked<'a> {
//     type Output = AlignedRoaringishPacked;
//     fn add(self, rhs: u32) -> Self::Output {
//         unsafe {
//             assume(self.doc_id_groups.len() == self.values.len());
//         }

//         let n = self.doc_id_groups.len() * 2;
//         let mut doc_id_groups: Box<[MaybeUninit<u64>], Aligned64> = Box::new_uninit_slice_in(n, Aligned64::default());
//         let mut values: Box<[u16], Aligned64> = unsafe { Box::new_zeroed_slice_in(n, Aligned64::default()).assume_init() };
//         let mut i = 0;

//         let current_mask = u16::MAX << rhs;
//         let next_mask = !current_mask;

//         let mut it = self.doc_id_groups.iter().zip(self.values.iter());
//         let Some((doc_id_group, packed_values)) = it.next() else {
//             return AlignedRoaringishPacked::default();
//         };

//         let new_values = packed_values.rotate_left(rhs);
//         let postions_current_group = new_values & current_mask;
//         let postions_new_group = new_values & next_mask;
//         if postions_current_group > 0 {
//             unsafe {
//                 doc_id_groups.get_unchecked_mut(i).write(*doc_id_group);
//                 *values.get_unchecked_mut(i) = postions_current_group;
//                 i += 1;
//             }
//         }
//         if postions_new_group > 0 {
//             unsafe {
//                 doc_id_groups.get_unchecked_mut(i).write(*doc_id_group + 1);
//                 *values.get_unchecked_mut(i) = postions_new_group;
//                 i += 1;
//             }
//         }

//         assert!(i > 0);

//         let it = it.array_chunks::<8>();

//         for (doc_id_group, packed_values) in it {
//             let new_values = packed_values.rotate_left(rhs);
//             let postions_current_group = new_values & current_mask;
//             let postions_new_group = new_values & next_mask;

//             // doc_id_group == highest_doc_id_group && postions_current_group > 0
//             // doc_id_group == highest_doc_id_group && postions_current_group == 0
//             //     dont increment i
//             //     i_doc_id_group = doc_id_group (not needed)
//             //     i_positions |= postions_current_group
//             // doc_id_group > highest_doc_id_group && postions_current_group > 0
//             //     increment i
//             //     i_doc_id_group = doc_id_group
//             //     i_positions |= postions_current_group
//             // doc_id_group > highest_doc_id_group && postions_current_group == 0
//             //     dont increment i
//             //     i_positions |= postions_current_group (not needed)

//             let highest_doc_id_group = unsafe { doc_id_groups.get_unchecked(i - 1).assume_init() };
//             i += (*doc_id_group > highest_doc_id_group && postions_current_group > 0) as usize;

//             let i_doc_id_group = unsafe { doc_id_groups.get_unchecked_mut(i - 1) };
//             let i_positions = unsafe { values.get_unchecked_mut(i - 1) };

//             // maybe transform this in a cmov ?
//             if postions_current_group > 0 {
//                 i_doc_id_group.write(*doc_id_group);
//             }
//             // unsafe {
//             //     let i_doc_id_group = std::mem::transmute::<_, &mut u64>(i_doc_id_group);
//             //     i_doc_id_group.cmovnz(doc_id_group, (postions_current_group > 0) as u8);
//             // }
//             *i_positions |= postions_current_group;

//             i += (postions_new_group > 0) as usize;
//             let i_doc_id_group = unsafe { doc_id_groups.get_unchecked_mut(i - 1) };
//             let i_positions = unsafe { values.get_unchecked_mut(i - 1) };

//             if postions_new_group > 0 {
//                 i_doc_id_group.write(*doc_id_group + 1);
//             }
//             // unsafe {
//             //     let i_doc_id_group = std::mem::transmute::<_, &mut u64>(i_doc_id_group);
//             //     i_doc_id_group.cmovnz(&(*doc_id_group + 1), (postions_new_group > 0) as u8);
//             // }
//             *i_positions |= postions_new_group;
//         }

//         unsafe {
//             AlignedRoaringishPacked {
//                 doc_id_groups: Vec::from_raw_parts_in(
//                     Box::into_raw(doc_id_groups) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//                 values: Vec::from_raw_parts_in(
//                     Box::into_raw(values) as *mut _,
//                     i,
//                     n,
//                     Aligned64::default(),
//                 ),
//             }
//         }
//     }
// }

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

impl<'a, A> Binary for BorrowRoaringishPacked<'a, A> {
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

impl<'a, A> Display for BorrowRoaringishPacked<'a, A> {
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

impl<'a> Arbitrary<'a> for RoaringishPacked {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.arbitrary_len::<(u64, NonZero<u16>)>()?;

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

        let it = u
            .arbitrary_iter::<NonZero<u16>>()?
            .take(doc_id_groups.len());
        let mut values: Vec<u16, Aligned64> = Vec::with_capacity_in(len, Aligned64::default());
        for v in it {
            values.push(v?.get());
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
