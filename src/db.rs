use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    fs::{DirEntry, OpenOptions},
    io::{Error, Write},
    iter::{Peekable, Zip},
    path::{Path, PathBuf},
    slice::Iter,
    str::FromStr,
    sync::atomic::AtomicU32,
};

use ahash::{AHashMap, AHashSet};
use fxhash::{FxHashMap, FxHashSet};
use heed::{
    byteorder::{BigEndian, LittleEndian},
    types::{Str, U32},
    Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
};
use rkyv::{
    ser::serializers::AllocSerializer, vec::ArchivedVec, Archive, Deserialize, Infallible,
    Serialize,
};
use roaring::RoaringBitmap;

use crate::{
    codecs::{NativeU32, ZeroCopyCodec},
    document::Document,
    indexer::Indexer,
    pl::{ArchivedPostingList, PostingList},
    utils::MAX_SEQ_LEN,
    Roaringish,
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
pub struct DB<D: Serialize<AllocSerializer<256>>> {
    pub(crate) env: Env,
    db_doc_id_to_document: Database<NativeU32, ZeroCopyCodec<D, 256>>,

    db_token_to_token_id: Database<Str, NativeU32>,
    db_token_id_to_posting_list: Database<NativeU32, ZeroCopyCodec<PostingList, 256>>,
    db_common_token_id_to_token: Database<NativeU32, Str>,

    common_tokens: AHashSet<Box<str>>,
}

impl<D: Serialize<AllocSerializer<256>> + 'static> DB<D> {
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
            .types::<NativeU32, ZeroCopyCodec<D, 256>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name("doc_id_to_document")
            .create(&mut wrtxn)
            .unwrap();

        let db_token_to_token_id = env
            .create_database(&mut wrtxn, Some("token_to_token_id"))
            .unwrap();

        let db_token_id_to_posting_list = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<PostingList, 256>>()
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
        }
    }

    pub fn indexer<'a, 'b>(&'b self, next_doc_id: &'a AtomicU32) -> Indexer<'a, 'b, D> {
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

    pub(crate) fn write_doc_id_to_document(
        &self,
        rwtxn: &mut RwTxn,
        doc_ids: Vec<u32>,
        documents: Vec<D>,
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
        token_id_to_pl: FxHashMap<u32, PostingList>,
    ) {
        let mut token_id_to_pl: Vec<_> = token_id_to_pl.into_iter().collect();
        token_id_to_pl.sort_unstable_by_key(|(token_id, _)| *token_id);

        for (token_id, pl) in token_id_to_pl {
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

impl<D> DB<D>
where
    D: Serialize<AllocSerializer<256>> + 'static,
    D::Archived: Deserialize<D, Infallible>,
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
            .types::<NativeU32, ZeroCopyCodec<D, 256>>()
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
            .types::<NativeU32, ZeroCopyCodec<PostingList, 256>>()
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

        let common_tokens = Self::read_common_tokens(&rotxn, db_common_token_id_to_token);

        rotxn.commit().unwrap();

        println!("Common tokens: {common_tokens:?}");

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
        db_common_token_id_to_token: Database<NativeU32, Str>,
    ) -> AHashSet<Box<str>> {
        db_common_token_id_to_token
            .iter(rotxn)
            .unwrap()
            .map(|r| r.unwrap().1.to_string().into_boxed_str())
            .collect()
    }

    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn single_token_search<T: AsRef<str>>(
        &self,
        rotxn: &RoTxn,
        tokens: &[T],
    ) -> Vec<u32> {
        let token = tokens.first().unwrap().as_ref();
        let Some(token_id) = self.db_token_to_token_id.get(&rotxn, token).unwrap() else {
            return Vec::new();
        };

        let pl = self.get_posting_list(rotxn, token_id);

        return pl.doc_ids.to_vec();
    }

    pub fn get_documents_by_ids(&self, doc_ids: Vec<u32>) -> Vec<D> {
        let rdtxn = self.env.read_txn().unwrap();
        doc_ids
            .into_iter()
            .map(|doc_id| {
                self.db_doc_id_to_document
                    .get(&rdtxn, &doc_id)
                    .unwrap()
                    .unwrap()
                    .deserialize(&mut rkyv::Infallible)
                    .unwrap()
            })
            .collect()
    }

    pub(crate) fn get_token_id(&self, rotxn: &RoTxn, token: &str) -> Option<u32> {
        unsafe {
            self.db_token_to_token_id
                .get(&rotxn, token)
                .unwrap_unchecked()
        }
    }

    #[inline(always)]
    pub(crate) fn get_posting_list<'a>(
        &self,
        rotxn: &'a RoTxn,
        token_id: u32,
    ) -> &'a ArchivedPostingList {
        unsafe {
            self.db_token_id_to_posting_list
                .get(&rotxn, &token_id)
                .unwrap_unchecked()
                .unwrap_unchecked()
        }
    }
}
