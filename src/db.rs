use std::{
    borrow::Cow, cell::UnsafeCell, cmp::Reverse, collections::BTreeSet, fmt::{Debug, Display}, fs::{DirEntry, OpenOptions}, io::{Error, Write}, iter::{Peekable, Zip}, path::{Path, PathBuf}, slice::Iter, str::FromStr, sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed}
};

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::{
    byteorder::{BigEndian, LittleEndian},
    types::{Str, U32},
    Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
};
use rkyv::{
    api::high::HighSerializer, ser::{allocator::ArenaHandle}, util::AlignedVec, vec::ArchivedVec, Archive, Deserialize, Serialize
};
use roaring::RoaringBitmap;

use crate::{
    codecs::{NativeU32, ZeroCopyCodec},
    normalize,
    pl::{ArchivedPostingList, CachePostingList, PostingList},
    roaringish::{intersect::Intersect, RoaringishPacked},
    tokenize,
    utils::MAX_SEQ_LEN,
};

#[derive(Default)]
pub struct Stats {
    pub normalize_tokenize: AtomicU64,
    pub merge: AtomicU64,
    pub get_pls: AtomicU64,
    pub first_gallop: AtomicU64,
    pub add_one_group: AtomicU64,
    pub second_gallop: AtomicU64,
    pub first_intersect: AtomicU64,
    pub second_intersect: AtomicU64,
    pub first_result: AtomicU64,
    pub second_result: AtomicU64,
    pub add_lhs: AtomicU64,
    pub add_rhs: AtomicU64,
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sum = 
            self.normalize_tokenize.load(Relaxed) +
            self.merge.load(Relaxed) +
            self.first_gallop.load(Relaxed) +
            self.add_one_group.load(Relaxed) +
            self.second_gallop.load(Relaxed) +
            self.first_intersect.load(Relaxed) +
            self.second_intersect.load(Relaxed) +
            self.first_result.load(Relaxed) +
            self.second_result.load(Relaxed) +
            self.add_lhs.load(Relaxed) +
            self.add_rhs.load(Relaxed);

        let sum_pl = sum + self.get_pls.load(Relaxed);

        let normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64 / sum as f64;
        let merge = self.merge.load(Relaxed) as f64 / sum as f64;
        // let get_pls = self.get_pls.load(Relaxed) as f64 / sum as f64;
        let first_gallop = self.first_gallop.load(Relaxed) as f64 / sum as f64;
        let add_one_group = self.add_one_group.load(Relaxed) as f64 / sum as f64;
        let second_gallop = self.second_gallop.load(Relaxed) as f64 / sum as f64;
        let first_intersect = self.first_intersect.load(Relaxed) as f64 / sum as f64;
        let second_intersect = self.second_intersect.load(Relaxed) as f64 / sum as f64;
        let first_result = self.first_result.load(Relaxed) as f64 / sum as f64;
        let second_result = self.second_result.load(Relaxed) as f64 / sum as f64;
        let add_lhs = self.add_lhs.load(Relaxed) as f64 / sum as f64;
        let add_rhs = self.add_rhs.load(Relaxed) as f64 / sum as f64;

        let pl_normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64 / sum_pl as f64;
        let pl_merge = self.merge.load(Relaxed) as f64 / sum_pl as f64;
        let pl_get_pls = self.get_pls.load(Relaxed) as f64 / sum_pl as f64;
        let pl_first_gallop = self.first_gallop.load(Relaxed) as f64 / sum_pl as f64;
        let pl_add_one_group = self.add_one_group.load(Relaxed) as f64 / sum_pl as f64;
        let pl_second_gallop = self.second_gallop.load(Relaxed) as f64 / sum_pl as f64;
        let pl_first_intersect = self.first_intersect.load(Relaxed) as f64 / sum_pl as f64;
        let pl_second_intersect = self.second_intersect.load(Relaxed) as f64 / sum_pl as f64;
        let pl_first_result = self.first_result.load(Relaxed) as f64 / sum_pl as f64;
        let pl_second_result = self.second_result.load(Relaxed) as f64 / sum_pl as f64;
        let pl_add_lhs = self.add_lhs.load(Relaxed) as f64 / sum_pl as f64;
        let pl_add_rhs = self.add_rhs.load(Relaxed) as f64 / sum_pl as f64;

        f.debug_struct("Stats")
            .field("normalize_tokenize", &format_args!("({:.3}ms, {normalize_tokenize:.3}%, {pl_normalize_tokenize:.3}%)", self.normalize_tokenize.load(Relaxed) as f64 / 1000f64))
            .field("merge", &format_args!("({:.3}ms, {merge:.3}%, {pl_merge:.3}%)", self.merge.load(Relaxed) as f64 / 1000f64))
            .field("get_pls", &format_args!("({:.3}ms, _%, {pl_get_pls:.3}%)", self.get_pls.load(Relaxed) as f64 / 1000f64))
            .field("first_gallop", &format_args!("({:.3}ms, {first_gallop:.3}%, {pl_first_gallop:.3}%)", self.first_gallop.load(Relaxed) as f64 / 1000f64))
            .field("add_one_group", &format_args!("({:.3}ms, {add_one_group:.3}%, {pl_add_one_group:.3}%)", self.add_one_group.load(Relaxed) as f64 / 1000f64))
            .field("second_gallop", &format_args!("({:.3}ms, {second_gallop:.3}%, {pl_second_gallop:.3}%)", self.second_gallop.load(Relaxed) as f64 / 1000f64))
            .field("first_intersect", &format_args!("({:.3}ms, {first_intersect:.3}%, {pl_first_intersect:.3}%)", self.first_intersect.load(Relaxed) as f64 / 1000f64))
            .field("second_intersect", &format_args!("({:.3}ms, {second_intersect:.3}%, {pl_second_intersect:.3}%)", self.second_intersect.load(Relaxed) as f64 / 1000f64))
            .field("first_result", &format_args!("({:.3}ms, {first_result:.3}%, {pl_first_result:.3}%)", self.first_result.load(Relaxed) as f64 / 1000f64))
            .field("second_result", &format_args!("({:.3}ms, {second_result:.3}%, {pl_second_result:.3}%)", self.second_result.load(Relaxed) as f64 / 1000f64))
            .field("add_lhs", &format_args!("({:.3}ms, {add_lhs:.3}%, {pl_add_lhs:.3}%)", self.add_lhs.load(Relaxed) as f64 / 1000f64))
            .field("add_rhs", &format_args!("({:.3}ms, {add_rhs:.3}%, {pl_add_rhs:.3}%)", self.add_rhs.load(Relaxed) as f64 / 1000f64))
            .finish()
    }
    // fn fmt(&self, f: &mut $crate::fmt::Formatter) -> $crate::fmt::Result {
    //     match self {
    //         Stats {
    //             normalize_tokenize: normalize_tokenize,
    //             merge: merge,
    //             get_pls: get_pls,
    //             first_gallop: first_gallop,
    //             add_one_group: add_one_group,
    //             second_gallop: second_gallop,
    //             first_intersect: first_intersect,
    //             second_intersect: second_intersect,
    //             first_result: first_result,
    //             second_result: second_result,
    //         } => f
    //             .debug_struct("Stats")
    //             .field("normalize_tokenize", &normalize_tokenize)
    //             .field("merge", &merge)
    //             .field("get_pls", &get_pls)
    //             .field("first_gallop", &first_gallop)
    //             .field("add_one_group", &add_one_group)
    //             .field("second_gallop", &second_gallop)
    //             .field("first_intersect", &first_intersect)
    //             .field("second_intersect", &second_intersect)
    //             .field("first_result", &first_result)
    //             .field("second_result", &second_result)
    //             .finish(),
    //     }
    // }
}

#[derive(Debug)]
pub struct ShardsInfo {
    pub indexed_files: AHashSet<PathBuf>,
    pub shards_with_error: AHashSet<PathBuf>,
    pub shards_ok: AHashMap<u32, u32>,
    pub next_shard_id: u32,
    pub next_doc_id: u32,
}

pub struct DB<D>
where
    D: for<'a> Serialize<HighSerializer<'a, AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>> + Archive,
{
    pub(crate) env: Env,
    db_doc_id_to_document: Database<NativeU32, ZeroCopyCodec<D>>,

    db_token_to_token_id: Database<Str, NativeU32>,
    db_token_id_to_posting_list: Database<NativeU32, ZeroCopyCodec<PostingList>>,
    db_common_token_id_to_token: Database<NativeU32, Str>,

    common_tokens: AHashSet<Box<str>>,
    common_tokens_id_to_posting_list: FxHashMap<u32, PostingList>,
}

unsafe impl<D> Send for DB<D> where
    D: for<'a> Serialize<HighSerializer<'a, AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>> + Archive + Send
{
}

unsafe impl<D> Sync for DB<D> where
    D: for<'a> Serialize<HighSerializer<'a, AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>> + Archive + Sync
{
}

impl<D> DB<D>
where
    D: for<'a> Serialize<HighSerializer<'a, AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    pub fn truncate(path: &Path, db_size: usize) -> Self {
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();

        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(5)
                .map_size(db_size)
                .flags(EnvFlags::WRITE_MAP | EnvFlags::MAP_ASYNC)
                .open(&path)
                .unwrap()
        };

        let mut wrtxn = env.write_txn().unwrap();

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("doc_id_to_document")
            .create(&mut wrtxn)
            .unwrap();

        let db_token_to_token_id = env
            .create_database(&mut wrtxn, Some("token_to_token_id"))
            .unwrap();

        let db_token_id_to_posting_list = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<PostingList>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("token_id_to_posting_list")
            .create(&mut wrtxn)
            .unwrap();

        let db_common_token_id_to_token = env
            .database_options()
            .types::<NativeU32, Str>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("common_token_id_to_token")
            .create(&mut wrtxn)
            .unwrap();

        wrtxn.commit().unwrap();

        Self {
            env,
            db_doc_id_to_document,
            db_token_id_to_posting_list,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens: AHashSet::new(),
            common_tokens_id_to_posting_list: FxHashMap::new()
        }
    }

    // pub fn indexer<'a>(&'a self, docs_per_shard: Option<u32>) -> Indexer<'a, D> {
    //     Indexer::new(self, docs_per_shard)
    // }

    pub(crate) fn write_doc_id_to_document(
        &self,
        rwtxn: &mut RwTxn,
        doc_ids: &[u32],
        documents: &[D],
    ) {
        for (doc_id, document) in doc_ids.iter().zip(documents.iter()) {
            self.db_doc_id_to_document
                .put_with_flags(rwtxn, PutFlags::APPEND, &doc_id, &document)
                .unwrap();
        }
    }

    pub(crate) fn write_token_to_token_id(
        &self,
        rwtxn: &mut RwTxn,
        token_to_token_id: &AHashMap<Box<str>, u32>,
    ) {
        let mut token_to_token_id: Vec<_> = token_to_token_id.iter().collect();
        token_to_token_id.sort_unstable_by(|(token0, _), (token1, _)| token0.cmp(token1));
        for (token, token_id) in token_to_token_id.iter() {
            self.db_token_to_token_id
                .put_with_flags(rwtxn, PutFlags::APPEND, &token, &token_id)
                .unwrap();
        }
    }

    pub(crate) fn write_postings_list(&self, rwtxn: &mut RwTxn, token_id_to_pl: &[PostingList]) {
        for (token_id, pl) in token_id_to_pl.iter().enumerate() {
            let token_id = token_id as u32;
            self.db_token_id_to_posting_list
                .put_with_flags(rwtxn, PutFlags::APPEND, &token_id, &pl)
                .unwrap();
        }
    }

    pub(crate) fn write_common_token_ids(
        &self,
        rwtxn: &mut RwTxn,
        common_token_ids: &FxHashSet<u32>,
        token_id_to_token: &[Box<str>],
    ) {
        let mut token_id_to_token: Vec<_> = common_token_ids
            .iter()
            .map(|token_id| (token_id, token_id_to_token[*token_id as usize].as_ref()))
            .collect();
        token_id_to_token.sort_unstable_by_key(|(token_id, _)| *token_id);

        for (token_id, token) in token_id_to_token.iter() {
            self.db_common_token_id_to_token
                .put_with_flags(rwtxn, PutFlags::APPEND, &token_id, token)
                .unwrap();
        }
    }
}

impl<D> DB<D>
where
    D: for<'a> Serialize<HighSerializer<'a, AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    pub fn open(path: &Path, db_size: usize) -> Self {
        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(5)
                .map_size(db_size)
                .flags(EnvFlags::READ_ONLY)
                .open(&path)
                .unwrap()
        };

        let rotxn = env.read_txn().unwrap();

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("doc_id_to_document")
            .open(&rotxn)
            .unwrap()
            .unwrap();

        let db_token_to_token_id = env
            .open_database(&rotxn, Some("token_to_token_id"))
            .unwrap()
            .unwrap();

        let db_token_id_to_posting_list = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<PostingList>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("token_id_to_posting_list")
            .open(&rotxn)
            .unwrap()
            .unwrap();

        let db_common_token_id_to_token = env
            .database_options()
            .types::<NativeU32, Str>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("common_token_id_to_token")
            .open(&rotxn)
            .unwrap()
            .unwrap();

        let (common_tokens, common_tokens_id_to_posting_list) = Self::read_common_tokens(&rotxn, db_common_token_id_to_token, db_token_id_to_posting_list);
        rotxn.commit().unwrap();

        Self {
            env,
            db_doc_id_to_document,
            db_token_id_to_posting_list,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens,
            common_tokens_id_to_posting_list
        }
    }

    fn read_common_tokens(
        rotxn: &RoTxn,
        db_common_token_id_to_token: Database<NativeU32, Str>,
        db_token_id_to_posting_list: Database<NativeU32, ZeroCopyCodec<PostingList>>
    ) -> (AHashSet<Box<str>>, FxHashMap<u32, PostingList>) {
        db_common_token_id_to_token
            .iter(rotxn)
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                let pl =
                    db_token_id_to_posting_list
                    .get(&rotxn, &r.0)
                    .unwrap()
                    .unwrap();
                let pl = rkyv::deserialize::<PostingList, rkyv::rancor::Error>(pl).unwrap();
                (r.1.to_string().into_boxed_str(), (r.0, pl))
            })
            .collect()
    }

    #[inline(always)]
    pub(crate) fn merge_tokens(&self, tokens: &[&str]) -> Vec<(String, u32)> {
        let mut final_tokens = Vec::with_capacity(tokens.len());
        let mut sequence = Vec::with_capacity(tokens.len());
        for token in tokens.iter() {
            if self.common_tokens.contains(*token) {
                sequence.push(*token);
                continue;
            }

            if sequence.len() <= 1 {
                final_tokens.extend(sequence.iter().map(|t| (t.to_string(), 1)));
                final_tokens.push((token.to_string(), 1));
                sequence.clear();
                continue;
            }

            for chunk in sequence.chunks(MAX_SEQ_LEN) {
                let token: String = chunk.into_iter().map(|t| *t).intersperse(" ").collect();
                final_tokens.push((token, chunk.len() as u32));
            }

            final_tokens.push((token.to_string(), 1));
            sequence.clear();
        }

        if sequence.len() > 1 {
            for chunk in sequence.chunks(MAX_SEQ_LEN) {
                let token: String = chunk.into_iter().map(|t| *t).intersperse(" ").collect();
                final_tokens.push((token, chunk.len() as u32));
            }
        } else {
            final_tokens.extend(sequence.iter().map(|t| (t.to_string(), 1)));
        }

        final_tokens
    }

    #[inline(always)]
    pub(crate) fn single_token_search(&self, rotxn: &RoTxn, token: &str) -> Vec<u32> {
        let Some(token_id) = self.db_token_to_token_id.get(&rotxn, token).unwrap() else {
            return Vec::new();
        };

        let pl = self.get_posting_list(rotxn, token_id);

        match pl {
            CachePostingList::Cached(pl) => pl.doc_ids.iter().copied().collect(),
            CachePostingList::NotCached(pl) => pl.doc_ids.iter().map(|doc_id| doc_id.to_native()).collect(),
        }
    }

    // pub fn get_documents_by_ids(&self, doc_ids: Vec<u32>)
    // where
    //     D: Display,
    // {
    //     // pub fn get_documents_by_ids(&self, doc_ids: Vec<u32>) -> Vec<D> {
    //     let rdtxn = self.env.read_txn().unwrap();
    //     for doc_id in doc_ids {
    //         let k = self
    //             .db_doc_id_to_document
    //             .get(&rdtxn, &doc_id)
    //             .unwrap()
    //             .unwrap()
    //             .deserialize(&mut rkyv::Infallible)
    //             .unwrap();
    //         println!("{k}");
    //         // panic!()
    //     }
    //     // doc_ids
    //     //     .into_iter()
    //     //     .map(|doc_id| {
    //     //         self.db_doc_id_to_document
    //     //             .get(&rdtxn, &doc_id)
    //     //             .unwrap()
    //     //             .unwrap()
    //     //             .deserialize(&mut rkyv::Infallible)
    //     //             .unwrap()
    //     //     })
    //     //     .collect()
    // }

    pub(crate) fn get_token_id(&self, rotxn: &RoTxn, token: &str) -> Option<u32> {
        unsafe {
            self.db_token_to_token_id
                .get(&rotxn, token)
                .unwrap_unchecked()
        }
    }

    #[inline(always)]
    pub(crate) fn get_posting_list<'a, 'b>(
        &'a self,
        rotxn: &'b RoTxn,
        token_id: u32,
    ) -> CachePostingList<'a, 'b> {
        match self.common_tokens_id_to_posting_list.get(&token_id) {
            Some(pl) => CachePostingList::Cached(pl),
            None => unsafe {
                CachePostingList::NotCached(self.db_token_id_to_posting_list
                    .get(&rotxn, &token_id)
                    .unwrap_unchecked()
                    .unwrap_unchecked())
            },
        }
    }

    pub fn search<I: Intersect>(&self, q: &str, stats: &Stats) -> Vec<u32> {
        let b = std::time::Instant::now();
        let q = normalize(q);
        let tokens: Vec<_> = tokenize(&q).collect();
        stats
            .normalize_tokenize
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if tokens.len() == 0 {
            return Vec::new();
        }

        let rotxn = self.env.read_txn().unwrap();
        if tokens.len() == 1 {
            return self.single_token_search(&rotxn, tokens.first().unwrap());
        }

        let b = std::time::Instant::now();
        let final_tokens = self.merge_tokens(&tokens);
        stats
            .merge
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
        if final_tokens.len() == 1 {
            return self.single_token_search(&rotxn, &final_tokens.first().unwrap().0);
        }

        let b = std::time::Instant::now();
        let deduped_tokens: AHashSet<_> = final_tokens.iter().map(|(t, _)| t).collect();
        let mut token_to_token_id = AHashMap::with_capacity(deduped_tokens.len());
        let mut token_id_to_packed = FxHashMap::with_capacity(deduped_tokens.len());
        for token in deduped_tokens.iter() {
            let Some(token_id) = self.get_token_id(&rotxn, token) else {
                return Vec::new();
            };

            let pl = self.get_posting_list(&rotxn, token_id);

            token_to_token_id.insert(*token, token_id);
            token_id_to_packed.insert(token_id, RoaringishPacked::from(pl));
        }
        stats
            .get_pls
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        let mut it = final_tokens.iter();

        let (lhs, lhs_len) = it.next().unwrap();
        let lhs = token_to_token_id.get(lhs).unwrap();
        let lhs = token_id_to_packed.get(lhs).unwrap();

        let (rhs, rhs_len) = it.next().unwrap();
        let rhs = token_to_token_id.get(rhs).unwrap();
        let rhs = token_id_to_packed.get(rhs).unwrap();
        let mut lhs = if *lhs_len > 1 {
            let b = std::time::Instant::now();
            let lhs = lhs + (*lhs_len - 1);
            stats
            .add_lhs
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
            lhs.intersect::<I>(rhs, *rhs_len, stats)
        } else {
            lhs.intersect::<I>(rhs, *rhs_len, stats)
        };

        for (t, t_len) in it {
            let rhs = token_to_token_id.get(t).unwrap();
            let rhs = token_id_to_packed.get(rhs).unwrap();
            lhs = lhs.intersect::<I>(rhs, *t_len, stats);
        }
        lhs.get_doc_ids()
    }
}
