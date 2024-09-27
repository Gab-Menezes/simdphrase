use std::{
    collections::BTreeSet,
    fmt::Debug,
    path::Path,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::{
    types::Str,
    Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec,
    Archive, Serialize,
};

use crate::{
    codecs::{NativeU32, ZeroCopyCodec},
    normalize,
    roaringish::{
        intersect::Intersect, ArchivedRoaringishPacked, Packed, RoaringishPacked,
        RoaringishPackedBuilder,
    },
    tokenize,
    utils::MAX_SEQ_LEN,
};

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

pub struct DB<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    pub(crate) env: Env,
    db_doc_id_to_document: Database<NativeU32, ZeroCopyCodec<D>>,

    db_token_to_token_id: Database<Str, NativeU32>,
    db_token_id_to_roaringish_packed: Database<NativeU32, ZeroCopyCodec<RoaringishPacked>>,
    db_common_token_id_to_token: Database<NativeU32, Str>,

    common_tokens: AHashSet<Box<str>>,
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
                .max_dbs(5)
                .map_size(db_size)
                .flags(EnvFlags::WRITE_MAP | EnvFlags::MAP_ASYNC)
                .open(path)
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

        let db_token_id_to_roaringish_packed = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<RoaringishPacked>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("db_token_id_to_roaringish_packed")
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
            db_token_id_to_roaringish_packed,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens: AHashSet::new(),
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

    pub(crate) fn write_token_to_token_id(
        &self,
        rwtxn: &mut RwTxn,
        token_to_token_id: &AHashMap<Box<str>, u32>,
    ) {
        let mut token_to_token_id: Vec<_> = token_to_token_id.iter().collect();
        token_to_token_id.sort_unstable_by(|(token0, _), (token1, _)| token0.cmp(token1));
        for (token, token_id) in token_to_token_id.iter() {
            self.db_token_to_token_id
                .put_with_flags(rwtxn, PutFlags::APPEND, token, token_id)
                .unwrap();
        }
    }

    pub(crate) fn write_postings_list(
        &self,
        rwtxn: &mut RwTxn,
        token_id_to_builder: Vec<RoaringishPackedBuilder>,
    ) {
        for (token_id, builder) in token_id_to_builder.into_iter().enumerate() {
            let token_id = token_id as u32;
            let packed = builder.finish();
            self.db_token_id_to_roaringish_packed
                .put_with_flags(rwtxn, PutFlags::APPEND, &token_id, &packed)
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
                .put_with_flags(rwtxn, PutFlags::APPEND, token_id, token)
                .unwrap();
        }
    }
}

impl<D> DB<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    pub fn open(path: &Path, db_size: usize) -> Self {
        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(5)
                .map_size(db_size)
                .flags(EnvFlags::READ_ONLY)
                .open(path)
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

        let db_token_id_to_roaringish_packed = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<RoaringishPacked>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("db_token_id_to_roaringish_packed")
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

        let common_tokens = Self::read_common_tokens(&rotxn, db_common_token_id_to_token);
        // let common_tokens = AHashSet::new();
        let sorted: BTreeSet<_> = common_tokens.iter().collect();
        rotxn.commit().unwrap();

        println!("Common tokens ({}): {sorted:?}", sorted.len());

        Self {
            env,
            db_doc_id_to_document,
            db_token_id_to_roaringish_packed,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens,
        }
    }

    fn read_common_tokens(
        rotxn: &RoTxn,
        db_common_token_id_to_token: Database<NativeU32, Str>,
    ) -> AHashSet<Box<str>> {
        db_common_token_id_to_token
            .iter(rotxn)
            .unwrap()
            .map(|r| r.unwrap().1.to_string().into_boxed_str())
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
                let token: String = chunk.iter().copied().intersperse(" ").collect();
                final_tokens.push((token, chunk.len() as u32));
            }

            final_tokens.push((token.to_string(), 1));
            sequence.clear();
        }

        if sequence.len() > 1 {
            for chunk in sequence.chunks(MAX_SEQ_LEN) {
                let token: String = chunk.iter().copied().intersperse(" ").collect();
                final_tokens.push((token, chunk.len() as u32));
            }
        } else {
            final_tokens.extend(sequence.iter().map(|t| (t.to_string(), 1)));
        }

        final_tokens
    }

    #[inline(always)]
    pub(crate) fn single_token_search(&self, rotxn: &RoTxn, token: &str) -> Vec<u32> {
        let Some(token_id) = self.db_token_to_token_id.get(rotxn, token).unwrap() else {
            return Vec::new();
        };

        self.get_roaringish_packed(rotxn, token_id).get_doc_ids()
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
                .get(rotxn, token)
                .unwrap_unchecked()
        }
    }

    #[inline(always)]
    pub(crate) fn get_roaringish_packed<'a>(
        &self,
        rotxn: &'a RoTxn,
        token_id: u32,
    ) -> &'a ArchivedRoaringishPacked {
        unsafe {
            self.db_token_id_to_roaringish_packed
                .get(rotxn, &token_id)
                .unwrap_unchecked()
                .unwrap_unchecked()
        }
    }

    pub fn search<I: Intersect>(&self, q: &str, stats: &Stats) -> Vec<u32> {
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

        let deduped_tokens: AHashSet<_> = final_tokens.iter().map(|(t, _)| t).collect();
        let mut token_to_token_id = AHashMap::with_capacity(deduped_tokens.len());
        let mut token_id_to_packed = FxHashMap::with_capacity(deduped_tokens.len());
        for token in deduped_tokens.iter() {
            let b = std::time::Instant::now();
            let Some(token_id) = self.get_token_id(&rotxn, token) else {
                return Vec::new();
            };

            let pl = self.get_roaringish_packed(&rotxn, token_id);
            stats
                .get_pls
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

            token_to_token_id.insert(*token, token_id);
            token_id_to_packed.insert(token_id, pl);
        }

        let mut it = final_tokens.iter();

        let (lhs, lhs_len) = it.next().unwrap();
        let lhs = token_to_token_id.get(lhs).unwrap();
        let lhs = token_id_to_packed.get(lhs).unwrap();

        let (rhs, rhs_len) = it.next().unwrap();
        let rhs = token_to_token_id.get(rhs).unwrap();
        let rhs = token_id_to_packed.get(rhs).unwrap();
        let mut lhs = if *lhs_len > 1 {
            let b = std::time::Instant::now();
            let lhs = *lhs + (*lhs_len - 1);
            stats
                .add_lhs
                .fetch_add(b.elapsed().as_micros() as u64, Relaxed);
            lhs.intersect::<I, _>(*rhs, *rhs_len, stats)
        } else {
            lhs.intersect::<I, _>(*rhs, *rhs_len, stats)
        };

        for (t, t_len) in it {
            let rhs = token_to_token_id.get(t).unwrap();
            let rhs = token_id_to_packed.get(rhs).unwrap();
            lhs = lhs.intersect::<I, _>(*rhs, *t_len, stats);
        }
        lhs.get_doc_ids()
    }
}
