use std::{cell::Cell, cmp::Reverse, collections::HashSet, path::Path};

use ahash::{AHashMap, AHashSet, HashMapExt, HashSetExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::RwTxn;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};

use crate::{
    db::{DB, N_GRAM_LEN},
    decreasing_window_iter::DecreasingWindows,
    roaringish::{RoaringishPacked, MAX_VALUE},
    utils::{normalize, tokenize},
};

#[derive(Debug)]
struct Batch<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    batch_id: u32,
    hllp_tokens: HyperLogLogPlus<Box<str>, ahash::RandomState>,

    next_token_id: u32,
    token_to_token_id: AHashMap<Box<str>, u32>,

    // this 2 containers are in sync
    token_id_to_roaringish_packed: Vec<RoaringishPacked>,
    token_id_to_token: Vec<Box<str>>,

    // this 3 containers are in sync
    doc_ids: Vec<u32>,
    documents: Vec<D>,
}

impl<D> Batch<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    fn new() -> Self {
        Self {
            batch_id: 0,
            hllp_tokens: HyperLogLogPlus::new(18, ahash::RandomState::new()).unwrap(),
            next_token_id: 0,
            token_to_token_id: AHashMap::new(),
            token_id_to_roaringish_packed: Vec::new(),
            token_id_to_token: Vec::new(),
            doc_ids: Vec::new(),
            documents: Vec::new(),
        }
    }

    fn estimate_number_of_distinct_tokens(&mut self) -> u64 {
        (self.hllp_tokens.count() * 1.015f64) as u64
    }

    fn clear(&mut self) {
        self.next_token_id = 0;
        self.token_to_token_id.clear();
        self.token_id_to_roaringish_packed.clear();
        self.token_id_to_token.clear();

        self.doc_ids.clear();
        self.documents.clear();
    }

    fn push(&mut self, doc_id: u32, content: &str, doc: D) {
        self.index_doc(content, doc_id);
        self.doc_ids.push(doc_id);
        self.documents.push(doc);
    }

    fn get_token_id(
        token: &str,
        hllp_tokens: &mut HyperLogLogPlus<Box<str>, ahash::RandomState>,
        token_to_token_id: &mut AHashMap<Box<str>, u32>,
        token_id_to_token: &mut Vec<Box<str>>,
        token_id_to_roaringish_packed: &mut Vec<RoaringishPacked>,
        next_token_id: &mut u32,
    ) -> u32 {
        hllp_tokens.insert(token);

        let (_, token_id) = token_to_token_id
            .raw_entry_mut()
            .from_key(token)
            .or_insert_with(|| {
                let current_token_id = *next_token_id;
                *next_token_id += 1;
                (token.to_string().into_boxed_str(), current_token_id)
            });

        if *token_id as usize >= token_id_to_token.len() {
            token_id_to_token.push(token.to_string().into_boxed_str());
            token_id_to_roaringish_packed.push(RoaringishPacked::default());
        }

        *token_id
    }

    fn index_doc(
        &mut self,
        content: &str,
        doc_id: u32,
    ) {
        let mut token_id_to_positions: FxHashMap<u32, Vec<u32>> = FxHashMap::new();
        let content = normalize(content);
        let tokens: Vec<_> = tokenize(&content).take(MAX_VALUE as usize).collect();
        let it = DecreasingWindows::new(&tokens, N_GRAM_LEN);
        for (pos, tokens) in it.enumerate() {
            for i in 1..(tokens.len()+1) {
                let token: String = tokens[..i].iter().copied().intersperse(" ").collect();
                let token_id = Self::get_token_id(
                    &token,
                    &mut self.hllp_tokens,
                    &mut self.token_to_token_id,
                    &mut self.token_id_to_token,
                    &mut self.token_id_to_roaringish_packed,
                    &mut self.next_token_id,
                );

                token_id_to_positions
                    .entry(token_id)
                    .or_default()
                    .push(pos as u32);
            }
        }

        for (token_id, positions) in token_id_to_positions.iter() {
            self.token_id_to_roaringish_packed[*token_id as usize].push(doc_id, positions);
        }
    }

    fn flush(
        &mut self,
        db: &DB<D>,
        rwtxn: &mut RwTxn,
        mmap_size: &mut usize,
    ) {
        if self.doc_ids.is_empty() {
            return;
        }

        let token_to_token_id = std::mem::take(&mut self.token_to_token_id);
        let token_id_to_roaringish_packed = std::mem::take(&mut self.token_id_to_roaringish_packed);

        db.write_token_to_roaringish_packed(
            token_to_token_id,
            token_id_to_roaringish_packed,
            mmap_size,
            self.batch_id,
        );
        db.write_doc_id_to_document(rwtxn, &self.doc_ids, &self.documents);

        self.batch_id += 1;
        self.clear();
    }
}

pub struct Indexer {
    batch_size: Option<u32>,
}

impl Indexer {
    pub fn new(batch_size: Option<u32>) -> Self {
        Self {
            batch_size,
        }
    }

    pub fn index<S, D, I>(&self, docs: I, path: &Path, db_size: usize) -> u32
    where
        S: AsRef<str>,
        I: IntoIterator<Item = (S, D)>,
        D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
            + Archive
            + 'static,
    {
        let db = DB::truncate(&path, db_size);
        let mut rwtxn = db.env.write_txn().unwrap();

        let mut batch = Batch::new();

        let batch_size = self.batch_size.unwrap_or(u32::MAX);

        let mut next_doc_id = 0;
        let mut mmap_size = 0;

        let mut b = std::time::Instant::now();
        for (content, doc) in docs.into_iter() {
            let doc_id = next_doc_id;
            next_doc_id += 1;

            batch.push(doc_id, content.as_ref(), doc);

            if next_doc_id % batch_size == 0 {
                batch.flush(&db, &mut rwtxn, &mut mmap_size);
                println!("flushed batch in {}", b.elapsed().as_secs());
                b = std::time::Instant::now();
            }
        }

        batch.flush(&db, &mut rwtxn, &mut mmap_size);
        println!("flushed batch in {}", b.elapsed().as_secs());

        let number_of_distinct_tokens = batch.estimate_number_of_distinct_tokens();

        b = std::time::Instant::now();
        db.generate_mmap_file(
            number_of_distinct_tokens,
            mmap_size,
            batch.batch_id,
            &mut rwtxn,
        );
        println!("write mmap {}", b.elapsed().as_secs());
        rwtxn.commit().unwrap();

        next_doc_id
    }
}
