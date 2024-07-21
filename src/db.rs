use std::{
    fs::{DirEntry, OpenOptions},
    io::{Error, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::AtomicU32,
};

use ahash::{AHashMap, AHashSet};
use fxhash::{FxHashMap, FxHashSet};
use heed::{
    byteorder::BigEndian,
    types::{Str, U32},
    Database, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
};
use roaring::RoaringBitmap;

use crate::{
    codecs::{BigEndianVariableOption, BincodeCodec},
    document::Document,
    indexer::Indexer,
    pl::{PostingList, PostingListCodec},
    utils::MAX_SEQ_LEN,
};

#[derive(Debug)]
pub struct ShardsInfo {
    pub indexed_files: AHashSet<PathBuf>,
    pub shards_with_error: AHashSet<PathBuf>,
    pub shards_ok: AHashMap<u32, u32>,
    pub next_shard_id: u32,
    pub next_doc_id: u32,
}

#[derive(Debug)]
pub struct DB {
    pub(crate) env: Env,
    db_doc_id_to_document:
        Database<U32<BigEndian>, BincodeCodec<Document, BigEndianVariableOption>>,

    db_token_to_token_id: Database<Str, U32<BigEndian>>,
    db_token_id_to_posting_list: Database<U32<BigEndian>, PostingListCodec>,
    db_common_token_id_to_token: Database<U32<BigEndian>, Str>,

    common_tokens: AHashSet<Box<str>>,
}

impl DB {
    pub fn truncate(path: &Path, shard_id: u32, db_size: usize) -> Self {
        let path = path.join(format!("shard_{shard_id}"));

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
            .create_database(&mut wrtxn, Some("doc_id_to_document"))
            .unwrap();

        let db_token_to_token_id: Database<Str, U32<BigEndian>> = env
            .create_database(&mut wrtxn, Some("token_to_token_id"))
            .unwrap();

        let db_token_id_to_posting_list = env
            .create_database(&mut wrtxn, Some("token_id_to_posting_list"))
            .unwrap();

        let db_common_token_id_to_token = env
            .create_database(&mut wrtxn, Some("common_token_id_to_token"))
            .unwrap();

        wrtxn.commit().unwrap();

        Self {
            env,
            db_doc_id_to_document,
            db_token_id_to_posting_list,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens: AHashSet::new(),
        }
    }

    pub fn indexer<'a, 'b>(&'b self, next_doc_id: &'a AtomicU32) -> Indexer<'a, 'b> {
        Indexer::new(next_doc_id, self)
    }

    pub fn get_shards_info(path: &Path) -> ShardsInfo {
        let mut indexed_files = AHashSet::new();
        let mut shards_ok = AHashMap::new();
        let mut shards_with_error = AHashSet::new();
        let mut next_shard_id = 0;
        let mut next_doc_id = 0;

        let Ok(dir) = std::fs::read_dir(path) else {
            return ShardsInfo {
                indexed_files,
                next_doc_id,
                next_shard_id,
                shards_with_error,
                shards_ok,
            };
        };

        let mut f = |shard_path: &Path| -> Option<(u32, u32)> {
            let shard_id = shard_path
                .file_name()?
                .to_str()?
                .strip_prefix("shard_")?
                .parse()
                .ok()?;
            next_shard_id = next_shard_id.max(shard_id);

            let file_name = shard_path.join("indexed_files.txt");
            let content = std::fs::read_to_string(&file_name).ok()?;
            let mut lines = content.lines();
            let docs_in_shard = lines.next()?.parse().ok()?;
            let last_doc_id = lines.next()?;

            next_doc_id = next_doc_id.max(last_doc_id.parse().ok()?);

            for l in lines {
                indexed_files.insert(PathBuf::from_str(l).ok()?);
            }

            Some((shard_id, docs_in_shard))
        };

        for shard_path in dir {
            let shard_path = shard_path.unwrap().path();
            match f(&shard_path) {
                Some((shard_id, docs_in_shard)) => {
                    shards_ok.insert(shard_id, docs_in_shard);
                }
                None => {
                    shards_with_error.insert(shard_path);
                }
            };
        }

        ShardsInfo {
            indexed_files,
            next_doc_id: next_doc_id + 1,
            next_shard_id: next_shard_id + 1,
            shards_with_error,
            shards_ok,
        }
    }

    pub(crate) fn write_files(&self, files: &[PathBuf], docs_in_shard: u32, last_doc_id: u32) {
        let path = self.env.path().join("indexed_files.txt");
        let mut f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&path)
            .unwrap();

        let mut final_str = docs_in_shard.to_string();
        final_str.push('\n');

        final_str.push_str(&last_doc_id.to_string());
        final_str.push('\n');
        for file in files {
            final_str.push_str(file.to_str().unwrap());
            final_str.push('\n');
        }

        f.write_all(final_str.as_bytes()).unwrap();
        f.flush().unwrap();
    }

    pub(crate) fn write_doc_id_to_document(
        &self,
        rwtxn: &mut RwTxn,
        doc_ids: Vec<u32>,
        documents: Vec<Document>,
    ) {
        for (doc_id, document) in doc_ids.into_iter().zip(documents.into_iter()) {
            self.db_doc_id_to_document
                .put_with_flags(rwtxn, PutFlags::APPEND, &doc_id, &document)
                .unwrap();
        }
    }

    pub(crate) fn write_token_to_token_id(
        &self,
        rwtxn: &mut RwTxn,
        token_to_token_id: AHashMap<Box<str>, u32>,
    ) {
        let mut token_to_token_id: Vec<_> = token_to_token_id.into_iter().collect();
        token_to_token_id.sort_unstable_by(|(token0, _), (token1, _)| token0.cmp(token1));
        for (token, token_id) in token_to_token_id {
            self.db_token_to_token_id
                .put_with_flags(rwtxn, PutFlags::APPEND, &token, &token_id)
                .unwrap();
        }
    }

    pub(crate) fn write_postings_list(
        &self,
        rwtxn: &mut RwTxn,
        token_id_doc_id_to_positions: FxHashMap<(u32, u32), RoaringBitmap>,
        number_of_tokens: usize,
    ) {
        let mut token_id_to_doc_ids_and_pos = vec![Vec::new(); number_of_tokens];
        for ((token_id, doc_id), pos) in token_id_doc_id_to_positions {
            token_id_to_doc_ids_and_pos[token_id as usize].push((doc_id, pos))
        }

        token_id_to_doc_ids_and_pos
            .iter_mut()
            .for_each(|doc_ids_and_pos| {
                doc_ids_and_pos.sort_unstable_by_key(|(doc_id, _)| *doc_id)
            });

        for (token_id, doc_ids_and_pos) in token_id_to_doc_ids_and_pos.into_iter().enumerate() {
            let token_id = token_id as u32;
            let mut doc_ids = RoaringBitmap::new();
            let mut positions = Vec::new();

            for (doc_id, pos) in doc_ids_and_pos {
                doc_ids.push(doc_id);
                positions.push(pos);
            }

            let pl = PostingList::new(doc_ids, positions);
            self.db_token_id_to_posting_list
                .put_with_flags(rwtxn, PutFlags::APPEND, &token_id, &pl)
                .unwrap();
        }
    }

    pub(crate) fn write_common_token_ids(
        &self,
        rwtxn: &mut RwTxn,
        common_token_ids: FxHashSet<u32>,
        token_id_to_token: &[Box<str>],
    ) {
        let mut token_id_to_token: Vec<_> = common_token_ids
            .into_iter()
            .map(|token_id| (token_id, token_id_to_token[token_id as usize].as_ref()))
            .collect();
        token_id_to_token.sort_unstable_by_key(|(token_id, _)| *token_id);

        for (token_id, token) in token_id_to_token {
            self.db_common_token_id_to_token
                .put_with_flags(rwtxn, PutFlags::APPEND, &token_id, token)
                .unwrap();
        }
    }
}

impl DB {
    pub fn open(path: &Path, shard_id: u32, db_size: usize) -> Self {
        let path = path.join(format!("shard_{shard_id}"));

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
            .open_database(&rotxn, Some("doc_id_to_document"))
            .unwrap()
            .unwrap();

        let db_token_to_token_id = env
            .open_database(&rotxn, Some("token_to_token_id"))
            .unwrap()
            .unwrap();

        let db_token_id_to_posting_list = env
            .open_database(&rotxn, Some("token_id_to_posting_list"))
            .unwrap()
            .unwrap();

        let db_common_token_id_to_token = env
            .open_database(&rotxn, Some("common_token_id_to_token"))
            .unwrap()
            .unwrap();

        let common_tokens = Self::read_common_tokens(&rotxn, db_common_token_id_to_token);

        rotxn.commit().unwrap();

        println!("Common tokens {shard_id}: {common_tokens:?}");

        Self {
            env,
            db_doc_id_to_document,
            db_token_id_to_posting_list,
            db_token_to_token_id,
            db_common_token_id_to_token,
            common_tokens,
        }
    }

    fn read_common_tokens(
        rotxn: &RoTxn,
        db_common_token_id_to_token: Database<U32<BigEndian>, Str>,
    ) -> AHashSet<Box<str>> {
        db_common_token_id_to_token
            .iter(rotxn)
            .unwrap()
            .map(|r| r.unwrap().1.to_string().into_boxed_str())
            .collect()
    }

    pub(crate) fn merge_tokens(&self, tokens: &[&str]) -> Vec<String> {
        let mut final_tokens = Vec::with_capacity(tokens.len());
        let mut sequence = Vec::with_capacity(tokens.len());
        for token in tokens.iter() {
            if self.common_tokens.contains(*token) {
                sequence.push(*token);
                continue;
            }

            if sequence.len() <= 1 {
                final_tokens.extend(sequence.iter().map(|t| t.to_string()));
                final_tokens.push(token.to_string());
                sequence.clear();
                continue;
            }

            for chunk in sequence.chunks(MAX_SEQ_LEN) {
                let token: String = chunk.into_iter().map(|t| *t).intersperse(" ").collect();
                final_tokens.push(token);
            }

            final_tokens.push(token.to_string());
            sequence.clear();
        }

        if sequence.len() > 1 {
            for chunk in sequence.chunks(MAX_SEQ_LEN) {
                let token: String = chunk.into_iter().map(|t| *t).intersperse(" ").collect();
                final_tokens.push(token);
            }
        } else {
            final_tokens.extend(sequence.iter().map(|t| t.to_string()));
        }

        final_tokens
    }

    pub(crate) fn begin_phrase_search(
        tokens: &[String],
        token_to_token_id: &AHashMap<&str, u32>,
        token_id_to_positions: &FxHashMap<u32, &RoaringBitmap>,
        query_len: usize,
    ) -> bool {
        let mut it = tokens.iter();

        let token = it.next().unwrap();
        let token_id = token_to_token_id.get(token.as_str()).unwrap();
        let positions = token_id_to_positions.get(token_id).unwrap();
        let u = positions.min().unwrap();
        let mut v = u;

        for token in it {
            let token_id = token_to_token_id.get(token.as_str()).unwrap();
            let positions = token_id_to_positions.get(token_id).unwrap();
            let rank = positions.rank(v) as u32;
            let Some(next) = positions.select(rank) else {
                return false;
            };
            v = next
        }

        if (v - u) as usize == query_len - 1 {
            true
        } else {
            Self::phrase_search(
                tokens,
                token_to_token_id,
                token_id_to_positions,
                v - query_len as u32,
                query_len,
            )
        }
    }

    #[inline(always)]
    fn phrase_search<'a, 'b: 'a>(
        tokens: &[String],
        token_to_token_id: &AHashMap<&str, u32>,
        token_id_to_positions: &FxHashMap<u32, &RoaringBitmap>,
        mut position: u32,
        query_len: usize,
    ) -> bool {
        loop {
            let mut it = tokens.iter();

            let token = it.next().unwrap();
            let token_id = token_to_token_id.get(token.as_str()).unwrap();
            let positions = token_id_to_positions.get(token_id).unwrap();
            let rank = positions.rank(position) as u32;
            let Some(next) = positions.select(rank) else {
                return false;
            };
            let u = next;
            let mut v = u;

            for token in it {
                let token_id = token_to_token_id.get(token.as_str()).unwrap();
                let positions = token_id_to_positions.get(token_id).unwrap();
                let rank = positions.rank(v) as u32;
                let Some(next) = positions.select(rank) else {
                    return false;
                };
                v = next
            }

            if (v - u) as usize == query_len - 1 {
                return true;
            }

            position = v - query_len as u32;
        }
    }

    pub(crate) fn single_token_search<T: AsRef<str>>(
        &self,
        rotxn: &RoTxn,
        tokens: &[T],
    ) -> RoaringBitmap {
        let token = tokens.first().unwrap().as_ref();
        let Some(token_id) = self.db_token_to_token_id.get(&rotxn, token).unwrap() else {
            return RoaringBitmap::new();
        };

        let pl = self
            .db_token_id_to_posting_list
            .get(&rotxn, &token_id)
            .unwrap()
            .unwrap();

        return pl.doc_ids;
    }

    pub fn get_documents_by_ids(&self, doc_ids: RoaringBitmap) -> Vec<String> {
        let rdtxn = self.env.read_txn().unwrap();
        doc_ids
            .into_iter()
            .map(|doc_id| {
                let doc = self
                    .db_doc_id_to_document
                    .get(&rdtxn, &doc_id)
                    .unwrap()
                    .unwrap();
                serde_json::to_string(&doc).unwrap()
            })
            .collect()
    }

    pub(crate) fn get_token_id(&self, rotxn: &RoTxn, token: &str) -> Option<u32> {
        self.db_token_to_token_id.get(&rotxn, token).unwrap()
    }

    pub(crate) fn get_posting_list(&self, rotxn: &RoTxn, token_id: u32) -> PostingList {
        self.db_token_id_to_posting_list
            .get(&rotxn, &token_id)
            .unwrap()
            .unwrap()
    }
}
