use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap, HashSet},
    fmt::Debug,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::{
    types::Str, Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
    Unspecified,
};
use memmap2::{Mmap, MmapMut};
use rkyv::{
    api::high::HighSerializer,
    boxed::ArchivedBox,
    deserialize,
    ser::{allocator::ArenaHandle, writer::IoWriter},
    util::AlignedVec,
    Archive, Archived, Deserialize, Serialize,
};

use crate::{
    codecs::{NativeU32, ZeroCopyCodec}, decreasing_window_iter::DecreasingWindows, normalize, roaringish::{
        intersect::Intersect, ArchivedRoaringishPacked, RoaringishPacked, RoaringishPackedKind,
    }, tokenize, utils::MAX_SEQ_LEN, AlignedBorrowRoaringishPacked
};

mod db_constants {
    pub const DB_DOC_ID_TO_DOCUMENT: &'static str = "doc_id_to_document";
    pub const DB_TOKEN_TO_OFFSETS: &'static str = "token_to_offsets";
    pub const KEY_COMMON_TOKENS: &'static str = "common_tokens";
    pub const FILE_ROARINGISH_PACKED: &'static str = "roaringish_packed";
    pub const TEMP_FILE_TOKEN_TO_PACKED: &'static str = "temp_token_to_packed";
}

#[derive(Default)]
pub struct Stats {
    pub normalize_tokenize: AtomicU64,
    pub merge: AtomicU64,
    pub get_pls: AtomicU64,
    pub first_intersect: AtomicU64,
    pub second_intersect: AtomicU64,
    pub first_result: AtomicU64,
    pub second_result: AtomicU64,
    pub add_lhs: AtomicU64,
    pub add_rhs: AtomicU64,
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sum = self.normalize_tokenize.load(Relaxed)
            + self.merge.load(Relaxed)
            + self.first_intersect.load(Relaxed)
            + self.second_intersect.load(Relaxed)
            + self.first_result.load(Relaxed)
            + self.second_result.load(Relaxed)
            + self.add_lhs.load(Relaxed)
            + self.add_rhs.load(Relaxed);

        let sum_pl = sum + self.get_pls.load(Relaxed);

        let normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64 / sum as f64;
        let merge = self.merge.load(Relaxed) as f64 / sum as f64;
        // let get_pls = self.get_pls.load(Relaxed) as f64 / sum as f64;
        let first_gallop = self.first_intersect.load(Relaxed) as f64 / sum as f64;
        let second_gallop = self.second_intersect.load(Relaxed) as f64 / sum as f64;
        let first_result = self.first_result.load(Relaxed) as f64 / sum as f64;
        let second_result = self.second_result.load(Relaxed) as f64 / sum as f64;
        let add_lhs = self.add_lhs.load(Relaxed) as f64 / sum as f64;
        let add_rhs = self.add_rhs.load(Relaxed) as f64 / sum as f64;

        let pl_normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64 / sum_pl as f64;
        let pl_merge = self.merge.load(Relaxed) as f64 / sum_pl as f64;
        let pl_get_pls = self.get_pls.load(Relaxed) as f64 / sum_pl as f64;
        let pl_first_gallop = self.first_intersect.load(Relaxed) as f64 / sum_pl as f64;
        let pl_second_gallop = self.second_intersect.load(Relaxed) as f64 / sum_pl as f64;
        let pl_first_result = self.first_result.load(Relaxed) as f64 / sum_pl as f64;
        let pl_second_result = self.second_result.load(Relaxed) as f64 / sum_pl as f64;
        let pl_add_lhs = self.add_lhs.load(Relaxed) as f64 / sum_pl as f64;
        let pl_add_rhs = self.add_rhs.load(Relaxed) as f64 / sum_pl as f64;

        f.debug_struct("Stats")
            .field(
                "normalize_tokenize",
                &format_args!(
                    "({:.3}ms, {normalize_tokenize:.3}%, {pl_normalize_tokenize:.3}%)",
                    self.normalize_tokenize.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "merge",
                &format_args!(
                    "({:.3}ms, {merge:.3}%, {pl_merge:.3}%)",
                    self.merge.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "get_pls",
                &format_args!(
                    "({:.3}ms, _%, {pl_get_pls:.3}%)",
                    self.get_pls.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "first_intersect",
                &format_args!(
                    "({:.3}ms, {first_gallop:.3}%, {pl_first_gallop:.3}%)",
                    self.first_intersect.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "second_intersect",
                &format_args!(
                    "({:.3}ms, {second_gallop:.3}%, {pl_second_gallop:.3}%)",
                    self.second_intersect.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "first_result",
                &format_args!(
                    "({:.3}ms, {first_result:.3}%, {pl_first_result:.3}%)",
                    self.first_result.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "second_result",
                &format_args!(
                    "({:.3}ms, {second_result:.3}%, {pl_second_result:.3}%)",
                    self.second_result.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "add_lhs",
                &format_args!(
                    "({:.3}ms, {add_lhs:.3}%, {pl_add_lhs:.3}%)",
                    self.add_lhs.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "add_rhs",
                &format_args!(
                    "({:.3}ms, {add_rhs:.3}%, {pl_add_rhs:.3}%)",
                    self.add_rhs.load(Relaxed) as f64 / 1000f64
                ),
            )
            .finish()
    }
}

#[derive(Debug, Serialize, Archive)]
struct Offset {
    begin: u64,
    doc_id_group_len: u64,
    values_len: u64,
}

pub struct DB<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    pub(crate) env: Env,
    db_main: Database<Unspecified, Unspecified>,
    db_doc_id_to_document: Database<NativeU32, ZeroCopyCodec<D>>,
    db_token_to_offsets: Database<Str, ZeroCopyCodec<Offset>>,
}

unsafe impl<D> Send for DB<D> where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Send
{
}

unsafe impl<D> Sync for DB<D> where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Sync
{
}

impl<D> DB<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    pub fn truncate(path: &Path, db_size: usize) -> Self {
        let _ = std::fs::remove_dir_all(path);
        std::fs::create_dir_all(path).unwrap();

        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(2)
                .map_size(db_size)
                .flags(EnvFlags::WRITE_MAP | EnvFlags::MAP_ASYNC)
                .open(path)
                .unwrap()
        };

        let mut wrtxn = env.write_txn().unwrap();

        let db_main = env.create_database(&mut wrtxn, None).unwrap();

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name(db_constants::DB_DOC_ID_TO_DOCUMENT)
            .create(&mut wrtxn)
            .unwrap();

        let db_token_to_offsets = env
            .create_database(&mut wrtxn, Some(db_constants::DB_TOKEN_TO_OFFSETS))
            .unwrap();

        wrtxn.commit().unwrap();

        Self {
            env,
            db_main,
            db_doc_id_to_document,
            db_token_to_offsets,
        }
    }

    pub(crate) fn write_doc_id_to_document(
        &self,
        rwtxn: &mut RwTxn,
        doc_ids: &[u32],
        documents: &[D],
    ) {
        for (doc_id, document) in doc_ids.iter().zip(documents.iter()) {
            self.db_doc_id_to_document
                .put_with_flags(rwtxn, PutFlags::APPEND, doc_id, document)
                .unwrap();
        }
    }

    pub(crate) fn write_token_to_roaringish_packed(
        &self,
        token_to_token_id: AHashMap<Box<str>, u32>,
        mut token_id_to_roaringish_packed: Vec<RoaringishPacked>,
        mmap_size: &mut usize,
        batch_id: u32,
    ) {
        let mut token_to_packed: Vec<_> = token_to_token_id
            .into_iter()
            .map(|(token, token_id)| {
                let packed = std::mem::take(&mut token_id_to_roaringish_packed[token_id as usize]);
                *mmap_size += packed.size_bytes();
                (token, packed)
            })
            .collect();
        token_to_packed.sort_unstable_by(|(token0, _), (token1, _)| token0.cmp(token1));

        let file_name = format!("{}_{batch_id}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
        let file = IoWriter::new(BufWriter::new(
            File::options()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(self.env.path().join(file_name))
                .unwrap(),
        ));
        rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(&token_to_packed, file).unwrap();
    }

    pub(crate) fn generate_mmap_file(
        &self,
        number_of_distinct_tokens: u64,
        mmap_size: usize,
        number_of_batches: u32,
        rwtxn: &mut RwTxn,
    ) {
        #[inline(always)]
        unsafe fn write_to_mmap<const N: usize>(
            mmap: &mut MmapMut,
            mmap_offset: &mut usize,
            bytes: &[u8],
        ) -> (usize, usize) {
            let ptr = mmap.as_ptr().add(*mmap_offset);
            let offset = ptr.align_offset(N);
            if *mmap_offset + offset >= mmap.len() {
                panic!("We fucked up, writing out of bounds");
            }
            *mmap_offset += offset;
            mmap[*mmap_offset..*mmap_offset + bytes.len()].copy_from_slice(bytes);

            let begin = *mmap_offset;
            *mmap_offset += bytes.len();
            (begin, bytes.len())
        }

        let file = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(self.env.path().join(db_constants::FILE_ROARINGISH_PACKED))
            .unwrap();
        file.set_len(mmap_size as u64 + (number_of_distinct_tokens * (64 + 16)))
            .unwrap();
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mut mmap_offset = 0;

        // we need to do this in 3 steps because of the borrow checker
        let files_mmaps: Vec<_> = (0..number_of_batches)
            .map(|i| {
                let file_name = format!("{}_{i}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
                let file = File::options()
                    .read(true)
                    .open(self.env.path().join(file_name))
                    .unwrap();
                unsafe { Mmap::map(&file).unwrap() }
            })
            .collect();
        let files_data: Vec<_> = files_mmaps
            .iter()
            .map(|mmap| unsafe {
                rkyv::access_unchecked::<Archived<Vec<(Box<str>, RoaringishPacked)>>>(mmap)
            })
            .collect();
        let mut iters: Vec<_> = files_data
            .iter()
            .map(|tokens_to_packeds| tokens_to_packeds.iter())
            .collect();

        struct ToMerge<'a> {
            token: &'a ArchivedBox<str>,
            packed: &'a ArchivedRoaringishPacked,
            i: usize,
        }
        impl<'a> PartialEq for ToMerge<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.token == other.token && self.i == other.i
            }
        }
        impl<'a> Eq for ToMerge<'a> {}
        impl<'a> PartialOrd for ToMerge<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                match self.token.partial_cmp(other.token) {
                    Some(std::cmp::Ordering::Equal) => self.i.partial_cmp(&other.i),
                    ord => ord,
                }
            }
        }
        impl<'a> Ord for ToMerge<'a> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match self.token.cmp(other.token) {
                    std::cmp::Ordering::Equal => self.i.cmp(&other.i),
                    ord => ord,
                }
            }
        }

        let mut heap = BinaryHeap::new();
        for (i, it) in iters.iter_mut().enumerate() {
            match it.next() {
                Some(token_to_packed) => heap.push(Reverse(ToMerge {
                    token: &token_to_packed.0,
                    packed: &token_to_packed.1,
                    i,
                })),
                None => {}
            }
        }

        while let Some(token_to_packed) = heap.pop() {
            let to_merge = token_to_packed.0;
            if let Some(token_to_packed) = (&mut iters[to_merge.i]).next() {
                heap.push(Reverse(ToMerge {
                    token: &token_to_packed.0,
                    packed: &token_to_packed.1,
                    i: to_merge.i,
                }));
            }

            let mut packed_kind = RoaringishPackedKind::Archived(&to_merge.packed);
            loop {
                let Some(next_to_merge) = heap.peek() else {
                    break;
                };

                if next_to_merge.0.token != to_merge.token {
                    break;
                }

                let next_to_merge = heap.pop().unwrap().0;
                if let Some(token_to_packed) = (&mut iters[next_to_merge.i]).next() {
                    heap.push(Reverse(ToMerge {
                        token: &token_to_packed.0,
                        packed: &token_to_packed.1,
                        i: next_to_merge.i,
                    }));
                }

                let next_to_merge_kind = RoaringishPackedKind::Archived(next_to_merge.packed);
                packed_kind = packed_kind.concat(next_to_merge_kind);
            }

            if to_merge.token.len() > 511 {
                continue;
            }

            let (doc_id_groups, values) = packed_kind.as_bytes();
            let offset = unsafe {
                let (begin, doc_id_group_len) =
                    write_to_mmap::<64>(&mut mmap, &mut mmap_offset, doc_id_groups);
                let (_, values_len) = write_to_mmap::<16>(&mut mmap, &mut mmap_offset, values);
                Offset {
                    begin: begin as u64,
                    doc_id_group_len: doc_id_group_len as u64,
                    values_len: values_len as u64,
                }
            };

            self.db_token_to_offsets
                .put_with_flags(rwtxn, PutFlags::APPEND, to_merge.token, &offset)
                .unwrap();
        }
    }

    fn read_common_tokens(
        rotxn: &RoTxn,
        db_main: Database<Unspecified, Unspecified>,
    ) -> HashSet<Box<str>> {
        let k = db_main
            .remap_types::<Str, ZeroCopyCodec<HashSet<Box<str>>>>()
            .get(rotxn, db_constants::KEY_COMMON_TOKENS)
            .unwrap()
            .unwrap();

        deserialize::<_, rkyv::rancor::Error>(k).unwrap()
    }

    pub(crate) fn write_common_tokens(&self, rwtxn: &mut RwTxn, common_tokens: &HashSet<Box<str>>) {
        self.db_main
            .remap_types::<Str, ZeroCopyCodec<HashSet<Box<str>>>>()
            .put(rwtxn, db_constants::KEY_COMMON_TOKENS, common_tokens)
            .unwrap();
    }

    pub fn open(path: &Path, db_size: usize) -> (Self, HashSet<Box<str>>, Mmap) {
        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(2)
                .map_size(db_size)
                .flags(EnvFlags::READ_ONLY)
                .open(path)
                .unwrap()
        };

        let rotxn = env.read_txn().unwrap();

        let db_main = env.open_database(&rotxn, None).unwrap().unwrap();

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name(db_constants::DB_DOC_ID_TO_DOCUMENT)
            .open(&rotxn)
            .unwrap()
            .unwrap();

        let db_token_to_offsets = env
            .open_database(&rotxn, Some(db_constants::DB_TOKEN_TO_OFFSETS))
            .unwrap()
            .unwrap();

        let common_tokens = Self::read_common_tokens(&rotxn, db_main);

        rotxn.commit().unwrap();

        let mmap_file = File::open(path.join(db_constants::FILE_ROARINGISH_PACKED)).unwrap();
        let mmap = unsafe { Mmap::map(&mmap_file).unwrap() };

        (
            Self {
                env,
                db_main,
                db_doc_id_to_document,
                db_token_to_offsets,
            },
            common_tokens,
            mmap,
        )
    }

    #[inline(always)]
    pub(crate) fn merge_common_tokens(
        &self,
        tokens: &[&str],
        common_tokens: &HashSet<Box<str>>,
    ) -> Vec<(String, u32)> {
        let mut final_tokens = Vec::with_capacity(tokens.len());
        let max_window_len = 3.try_into().unwrap();
        let mut it = DecreasingWindows::new(tokens, max_window_len);
        while let Some(tokens) = it.next() {
            let mut i = tokens[1..].iter().take_while(|t| common_tokens.contains(**t)).count();
            if common_tokens.contains(tokens[0]) {
                i += 1;
            }
            let len = (i + 1).min(tokens.len());
            let token: String = tokens[..len].iter().copied().intersperse(" ").collect();
            final_tokens.push((token, len as u32));
            for _ in 0..(len - 1) {
                let _ = it.next();
            }
        }
        final_tokens
    }

    #[inline(always)]
    pub(crate) fn get_roaringish_packed<'a>(
        &self,
        rotxn: &RoTxn,
        token: &str,
        mmap: &'a Mmap,
    ) -> Option<AlignedBorrowRoaringishPacked<'a>> {
        let offset = self.db_token_to_offsets.get(rotxn, token).unwrap()?;

        let begin = offset.begin.to_native() as usize;
        let end = begin + offset.doc_id_group_len.to_native() as usize;
        let (l, doc_id_groups, r) = unsafe { &mmap[begin..end].align_to::<u64>() };
        assert!(l.is_empty());
        assert!(r.is_empty());

        let values_offset = unsafe { mmap.as_ptr().add(end).align_offset(16) };

        let offset_advise = begin;
        let len_advise = offset.doc_id_group_len.to_native() as usize + values_offset + offset.values_len.to_native() as usize;

        if end + values_offset >= mmap.len() {
            return None;
        }
        let begin = end + values_offset;
        let end = begin + offset.values_len.to_native() as usize;
        let (l, values, r) = unsafe { &mmap[begin..end].align_to::<u16>() };
        assert!(l.is_empty());
        assert!(r.is_empty());

        mmap.advise_range(memmap2::Advice::Sequential, offset_advise, len_advise).unwrap();

        Some(AlignedBorrowRoaringishPacked::new_raw(
            doc_id_groups,
            values,
        ))
    }

    pub fn search<I: Intersect>(
        &self,
        q: &str,
        stats: &Stats,
        common_tokens: &HashSet<Box<str>>,
        mmap: &Mmap,
    ) -> Vec<u32> {
        let b = std::time::Instant::now();
        let q = normalize(q);
        let tokens: Vec<_> = tokenize(&q).collect();
        stats
            .normalize_tokenize
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if tokens.is_empty() {
            return Vec::new();
        }

        let rotxn = self.env.read_txn().unwrap();
        if tokens.len() == 1 {
            return self
                .get_roaringish_packed(&rotxn, tokens.first().unwrap(), mmap)
                .map(|p| p.get_doc_ids())
                .unwrap_or_default();
        }

        let b = std::time::Instant::now();
        let final_tokens = self.merge_common_tokens(&tokens, common_tokens);
        stats
            .merge
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
        if final_tokens.len() == 1 {
            return self
                .get_roaringish_packed(&rotxn, &final_tokens.first().unwrap().0, mmap)
                .map(|p| p.get_doc_ids())
                .unwrap_or_default();
        }

        let deduped_tokens: AHashSet<_> = final_tokens.iter().map(|(t, _)| t).collect();
        let mut token_to_packed = AHashMap::with_capacity(deduped_tokens.len());
        for token in deduped_tokens.iter() {
            let b = std::time::Instant::now();
            let Some(packed) = self.get_roaringish_packed(&rotxn, token, mmap) else {
                return Vec::new();
            };
            token_to_packed.insert(*token, packed);

            stats
                .get_pls
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
        }

        let mut it = final_tokens.iter();

        let (lhs, lhs_len) = it.next().unwrap();
        let lhs = token_to_packed.get(lhs).unwrap();

        let (rhs, rhs_len) = it.next().unwrap();
        let rhs = token_to_packed.get(rhs).unwrap();

        let mut lhs = if *lhs_len > 1 {
            let b = std::time::Instant::now();
            let lhs = lhs + (*lhs_len - 1);
            stats
                .add_lhs
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

            let lhs = AlignedBorrowRoaringishPacked::new(&lhs);
            lhs.intersect::<I>(rhs, *rhs_len, stats)
        } else {
            lhs.intersect::<I>(rhs, *rhs_len, stats)
        };
        let mut borrow_lhs = AlignedBorrowRoaringishPacked::new(&lhs);

        for (t, t_len) in it {
            let rhs = token_to_packed.get(t).unwrap();
            lhs = borrow_lhs.intersect::<I>(rhs, *t_len, stats);
            borrow_lhs = AlignedBorrowRoaringishPacked::new(&lhs);
        }

        borrow_lhs = AlignedBorrowRoaringishPacked::new(&lhs);
        borrow_lhs.get_doc_ids()
    }
}
