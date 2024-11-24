use std::{
    borrow::Borrow,
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet},
    fmt::Debug,
    fs::File,
    io::BufWriter,
    num::NonZero,
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    u32,
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
    boxed::{ArchivedBox, BoxResolver},
    deserialize,
    rancor::Fallible,
    ser::{allocator::ArenaHandle, writer::IoWriter, Allocator, Serializer, Writer},
    tuple::{ArchivedTuple2, ArchivedTuple3},
    util::AlignedVec,
    vec::{ArchivedVec, VecResolver},
    with::{Inline, InlineAsBox},
    Archive, Archived, Deserialize, Place, Serialize,
};

use crate::{
    codecs::{NativeU32, ZeroCopyCodec},
    decreasing_window_iter::DecreasingWindows,
    normalize,
    roaringish::{
        intersect::Intersect, Aligned, ArchivedBorrowRoaringishPacked, RoaringishPackedKind,
        Unaligned,
    },
    tokenize, BorrowRoaringishPacked, RoaringishPacked,
};

#[derive(Archive, Serialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BorrowStr<'a>(#[rkyv(with = InlineAsBox)] &'a str);

mod db_constants {
    pub const DB_DOC_ID_TO_DOCUMENT: &'static str = "doc_id_to_document";
    pub const DB_TOKEN_TO_OFFSETS: &'static str = "token_to_offsets";
    pub const KEY_COMMON_TOKENS: &'static str = "common_tokens";
    pub const FILE_ROARINGISH_PACKED: &'static str = "roaringish_packed";
    pub const TEMP_FILE_TOKEN_TO_PACKED: &'static str = "temp_token_to_packed";
}

pub const MAX_WINDOW_LEN: NonZero<usize> = unsafe { NonZero::new_unchecked(3) };

#[derive(Default)]
pub struct Stats {
    pub normalize_tokenize: AtomicU64,
    pub merge: AtomicU64,
    pub first_binary_search: AtomicU64,
    pub first_intersect: AtomicU64,
    pub first_intersect_simd: AtomicU64,
    pub first_intersect_naive: AtomicU64,

    pub second_binary_search: AtomicU64,
    pub second_intersect: AtomicU64,
    pub second_intersect_simd: AtomicU64,
    pub second_intersect_naive: AtomicU64,

    pub first_result: AtomicU64,
    pub second_result: AtomicU64,
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sum = self.normalize_tokenize.load(Relaxed)
            + self.merge.load(Relaxed)
            + self.first_binary_search.load(Relaxed)
            + self.first_intersect.load(Relaxed)
            + self.second_binary_search.load(Relaxed)
            + self.second_intersect.load(Relaxed)
            + self.first_result.load(Relaxed)
            + self.second_result.load(Relaxed);

        let normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64 / sum as f64 * 100f64;
        let merge = self.merge.load(Relaxed) as f64 / sum as f64 * 100f64;
        let first_binary_search = self.first_binary_search.load(Relaxed) as f64 / sum as f64 * 100f64;
        let first_intersect = self.first_intersect.load(Relaxed) as f64 / sum as f64 * 100f64;
        let second_binary_search = self.second_binary_search.load(Relaxed) as f64 / sum as f64 * 100f64;
        let second_intersect = self.second_intersect.load(Relaxed) as f64 / sum as f64 * 100f64;
        let first_result = self.first_result.load(Relaxed) as f64 / sum as f64 * 100f64;
        let second_result = self.second_result.load(Relaxed) as f64 / sum as f64 * 100f64;

        f.debug_struct("Stats")
            .field(
                "normalize_tokenize",
                &format_args!(
                    "({:.3}ms, {normalize_tokenize:.3}%)",
                    self.normalize_tokenize.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "merge",
                &format_args!(
                    "({:.3}ms, {merge:.3}%)",
                    self.merge.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "first_binary_search",
                &format_args!(
                    "({:.3}ms, {first_binary_search:.3}%)",
                    self.first_binary_search.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "first_intersect",
                &format_args!(
                    "({:.3}ms, {first_intersect:.3}%)",
                    self.first_intersect.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "    first_intersect_simd",
                &format_args!(
                    "({:.3}ms)",
                    self.first_intersect_simd.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "    first_intersect_naive",
                &format_args!(
                    "({:.3}ms)",
                    self.first_intersect_naive.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "second_binary_search",
                &format_args!(
                    "({:.3}ms, {second_binary_search:.3}%)",
                    self.second_binary_search.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "second_intersect",
                &format_args!(
                    "({:.3}ms, {second_intersect:.3}%)",
                    self.second_intersect.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "    second_intersect_simd",
                &format_args!(
                    "({:.3}ms)",
                    self.second_intersect_simd.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "    second_intersect_naive",
                &format_args!(
                    "({:.3}ms)",
                    self.second_intersect_naive.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "first_result",
                &format_args!(
                    "({:.3}ms, {first_result:.3}%)",
                    self.first_result.load(Relaxed) as f64 / 1000f64
                ),
            )
            .field(
                "second_result",
                &format_args!(
                    "({:.3}ms, {second_result:.3}%)",
                    self.second_result.load(Relaxed) as f64 / 1000f64
                ),
            )
            .finish()
    }
}

#[derive(Debug, Serialize, Archive)]
struct Offset {
    begin: u64,
    len: u64,
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
        token_to_token_id: &AHashMap<Box<str>, u32>,
        token_id_to_roaringish_packed: &[RoaringishPacked],
        mmap_size: &mut usize,
        batch_id: u32,
    ) {
        let mut token_to_packed: Vec<_> = token_to_token_id
            .into_iter()
            .map(|(token, token_id)| {
                let packed = &token_id_to_roaringish_packed[*token_id as usize];
                *mmap_size += packed.size_bytes();
                (BorrowStr(token), BorrowRoaringishPacked::new(packed))
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
        ) -> Offset {
            let ptr = mmap.as_ptr().add(*mmap_offset);
            let offset = ptr.align_offset(N);

            *mmap_offset += offset;
            mmap[*mmap_offset..*mmap_offset + bytes.len()].copy_from_slice(bytes);

            let begin = *mmap_offset;
            *mmap_offset += bytes.len();
            Offset {
                begin: begin as u64,
                len: bytes.len() as u64,
            }
        }

        let file = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(self.env.path().join(db_constants::FILE_ROARINGISH_PACKED))
            .unwrap();
        file.set_len(mmap_size as u64 + (number_of_distinct_tokens * 64))
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
                rkyv::access_unchecked::<
                    Archived<Vec<(BorrowStr<'_>, BorrowRoaringishPacked<'_, Unaligned>)>>,
                >(mmap)
            })
            .collect();
        let mut iters: Vec<_> = files_data
            .iter()
            .map(|tokens_to_packeds| tokens_to_packeds.iter())
            .collect();

        struct ToMerge<'a> {
            token: &'a ArchivedBorrowStr<'a>,
            packed: &'a ArchivedBorrowRoaringishPacked<'a, Unaligned>,
            i: usize,
        }
        impl<'a> PartialEq for ToMerge<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.token.0 == other.token.0 && self.i == other.i
            }
        }
        impl<'a> Eq for ToMerge<'a> {}
        impl<'a> PartialOrd for ToMerge<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                match self.token.0.partial_cmp(&other.token.0) {
                    Some(std::cmp::Ordering::Equal) => self.i.partial_cmp(&other.i),
                    ord => ord,
                }
            }
        }
        impl<'a> Ord for ToMerge<'a> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match self.token.0.cmp(&other.token.0) {
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

            let mut packed_kind = RoaringishPackedKind::Archived(to_merge.packed);
            loop {
                let Some(next_to_merge) = heap.peek() else {
                    break;
                };

                if next_to_merge.0.token.0 != to_merge.token.0 {
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

            if to_merge.token.0.len() > 511 {
                continue;
            }

            let packed = packed_kind.as_bytes();
            let offset = unsafe { write_to_mmap::<64>(&mut mmap, &mut mmap_offset, packed) };
            self.db_token_to_offsets
                .put_with_flags(rwtxn, PutFlags::APPEND, &to_merge.token.0, &offset)
                .unwrap();
        }

        drop(iters);
        drop(files_data);
        drop(files_mmaps);

        for i in 0..number_of_batches {
            let file_name = format!("{}_{i}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
            std::fs::remove_file(self.env.path().join(file_name)).unwrap();
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
    pub(crate) fn merge_and_minimize_tokens<'a, 'b, 'c>(
        &self,
        rotxn: &RoTxn,
        tokens: &'b [&'c str],
        common_tokens: &HashSet<Box<str>>,
        mmap: &'a Mmap,
    ) -> (
        Vec<&'b [&'c str]>,
        AHashMap<&'b [&'c str], BorrowRoaringishPacked<'a, Aligned>>,
    ) {
        // TODO: improve this, temporary code just to make sure things working
        // maybe use smallvec ?
        fn inner_merge_and_minimize_tokens<'a, 'b, 'c, D>(
            me: &DB<D>,
            rotxn: &RoTxn,
            tokens: &'b [&'c str],
            common_tokens: &HashSet<Box<str>>,
            token_to_packed: &mut AHashMap<&'b [&'c str], BorrowRoaringishPacked<'a, Aligned>>,
            mmap: &'a Mmap,
            memo_token_to_score_choices: &mut AHashMap<&'b [&'c str], (usize, Vec<&'b [&'c str]>)>,
        ) -> usize
        where
            D: for<'d> Serialize<HighSerializer<AlignedVec, ArenaHandle<'d>, rkyv::rancor::Error>>
                + Archive
                + 'static,
        {
            if let Some(r) = memo_token_to_score_choices.get(tokens) {
                return r.0;
            }

            if tokens.len() == 1 {
                let score = match token_to_packed.get(tokens) {
                    Some(packed) => packed.len(),
                    None => {
                        let concatened: String =
                            tokens.into_iter().copied().intersperse(" ").collect();
                        match me.get_roaringish_packed(rotxn, &concatened, mmap) {
                            Some(packed) => {
                                let score = packed.len();
                                token_to_packed.insert(tokens, packed);
                                score
                            }
                            None => return 0,
                        }
                    }
                };
                memo_token_to_score_choices.insert(tokens, (score, vec![tokens]));
                return score;
            }

            let mut final_score = usize::MAX;
            let mut choices = Vec::new();

            // TODO: fix this, it looks ugly
            let mut end = tokens[1..]
                .into_iter()
                .take(MAX_WINDOW_LEN.get() - 1)
                .take_while(|t| common_tokens.contains(**t))
                .count()
                + 2;
            if common_tokens.contains(tokens[0]) {
                end += 1;
            }
            end = end.min(MAX_WINDOW_LEN.get() + 1).min(tokens.len() + 1);

            for i in (1..end).rev() {
                let (tokens, rem) = tokens.split_at(i);

                let score = match token_to_packed.get(tokens) {
                    Some(packed) => packed.len(),
                    None => {
                        let concatened: String =
                            tokens.into_iter().copied().intersperse(" ").collect();
                        match me.get_roaringish_packed(rotxn, &concatened, mmap) {
                            Some(packed) => {
                                let score = packed.len();
                                token_to_packed.insert(tokens, packed);
                                score
                            }
                            None => return 0,
                        }
                    }
                };

                let mut rem_score = 0;
                if !rem.is_empty() {
                    rem_score = inner_merge_and_minimize_tokens(
                        me,
                        rotxn,
                        rem,
                        common_tokens,
                        token_to_packed,
                        mmap,
                        memo_token_to_score_choices,
                    );
                    if rem_score == 0 {
                        return 0;
                    }
                }

                let calc_score = score + rem_score;
                if calc_score < final_score {
                    final_score = calc_score;
                    choices.clear();
                    choices.push(tokens);
                    if let Some((_, rem_choices)) = memo_token_to_score_choices.get(rem) {
                        choices.extend(rem_choices.into_iter());
                    };
                }
            }

            memo_token_to_score_choices.insert(tokens, (final_score, choices));
            final_score
        }

        if common_tokens.is_empty() {
            let mut choices = Vec::with_capacity(tokens.len());
            let mut token_to_packed = AHashMap::with_capacity(tokens.len());
            for i in 0..tokens.len() {
                match self.get_roaringish_packed(rotxn, tokens[i], mmap) {
                    Some(packed) => token_to_packed.insert(tokens, packed),
                    None => return (Vec::new(), AHashMap::new()),
                };
                choices.push(&tokens[i..i + 1]);
            }
            return (choices, token_to_packed);
        }

        let n = MAX_WINDOW_LEN.get();
        let l = tokens.len();
        let len = n * (l.max(n) - n + 1) + ((n - 1) * n) / 2;
        let mut memo_token_to_score_choices = AHashMap::with_capacity(len);
        let mut token_to_packed = AHashMap::with_capacity(len);

        let score = inner_merge_and_minimize_tokens(
            self,
            rotxn,
            tokens,
            common_tokens,
            &mut token_to_packed,
            mmap,
            &mut memo_token_to_score_choices,
        );
        if score == 0 {
            return (Vec::new(), AHashMap::new());
        }
        match memo_token_to_score_choices.remove(tokens) {
            Some((_, choices)) => (choices, token_to_packed),
            None => (Vec::new(), AHashMap::new()),
        }
    }

    fn get_roaringish_packed_from_offset<'a>(
        offset: &ArchivedOffset,
        mmap: &'a Mmap,
    ) -> Option<BorrowRoaringishPacked<'a, Aligned>> {
        let begin = offset.begin.to_native() as usize;
        let len = offset.len.to_native() as usize;
        let end = begin + len;
        let (l, packed, r) = unsafe { &mmap[begin..end].align_to::<u64>() };
        assert!(l.is_empty());
        assert!(r.is_empty());

        mmap.advise_range(memmap2::Advice::Sequential, begin, len)
            .unwrap();

        Some(BorrowRoaringishPacked::new_raw(packed))
    }

    #[inline(always)]
    pub(crate) fn get_roaringish_packed<'a>(
        &self,
        rotxn: &RoTxn,
        token: &str,
        mmap: &'a Mmap,
    ) -> Option<BorrowRoaringishPacked<'a, Aligned>> {
        let offset = self.db_token_to_offsets.get(rotxn, token).unwrap()?;
        Self::get_roaringish_packed_from_offset(offset, mmap)
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
        let (final_tokens, token_to_packed) =
            self.merge_and_minimize_tokens(&rotxn, &tokens, &common_tokens, mmap);
        stats
            .merge
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if final_tokens.is_empty() {
            return Vec::new();
        }

        if final_tokens.len() == 1 {
            return token_to_packed
                .get(final_tokens.first().unwrap())
                .unwrap()
                .get_doc_ids();
        }

        let mut min = usize::MAX;
        let mut i = usize::MAX;
        for (j, ts) in final_tokens.windows(2).enumerate() {
            let l0 = token_to_packed.get(&ts[0]).unwrap().len();
            let l1 = token_to_packed.get(&ts[1]).unwrap().len();
            let l = l0 + l1;
            if l <= min {
                i = j;
                min = l;
            }
        }

        let lhs = &final_tokens[i];
        let mut lhs_len = lhs.len() as u32;
        let lhs = token_to_packed.get(lhs).unwrap();

        let rhs = &final_tokens[i+1];
        let mut rhs_len = rhs.len() as u32;
        let rhs = token_to_packed.get(rhs).unwrap();

        let mut result = lhs.intersect::<I>(*rhs, lhs_len, stats);
        let mut result_borrow = BorrowRoaringishPacked::new(&result);

        let mut left_i = i.wrapping_sub(1);
        let mut right_i = i + 2;

        loop {
            let lhs = final_tokens.get(left_i);
            let rhs = final_tokens.get(right_i);
            match (lhs, rhs) {
                (Some(t_lhs), Some(t_rhs)) => {
                    let lhs = token_to_packed.get(t_lhs).unwrap();
                    let rhs = token_to_packed.get(t_rhs).unwrap();
                    if lhs.len() <= rhs.len() {
                        lhs_len += t_lhs.len() as u32;

                        result = lhs.intersect::<I>(result_borrow, lhs_len, stats);
                        result_borrow = BorrowRoaringishPacked::new(&result);

                        left_i = left_i.wrapping_sub(1);
                    } else {
                        result = result_borrow.intersect::<I>(*rhs, rhs_len, stats);
                        result_borrow = BorrowRoaringishPacked::new(&result);
    
                        lhs_len += rhs_len;
                        rhs_len = t_rhs.len() as u32;

                        right_i += 1;
                    }
                },
                (Some(t_lhs), None) => {
                    let lhs = token_to_packed.get(t_lhs).unwrap();
                    lhs_len += t_lhs.len() as u32;

                    result = lhs.intersect::<I>(result_borrow, lhs_len, stats);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    left_i = left_i.wrapping_sub(1);
                },
                (None, Some(t_rhs)) => {
                    let rhs = token_to_packed.get(t_rhs).unwrap();

                    result = result_borrow.intersect::<I>(*rhs, rhs_len, stats);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    lhs_len += rhs_len;
                    rhs_len = t_rhs.len() as u32;

                    right_i += 1;
                },
                (None, None) => break,
            }
        }

        result_borrow.get_doc_ids()
    }
}
