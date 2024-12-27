use std::{cell::Cell, cmp::Reverse, collections::HashSet, path::Path};

use gxhash::{HashMap as GxHashMap, HashSet as GxHashSet, HashMapExt, HashSetExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::RwTxn;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};

use crate::{
    db::{DB, MAX_WINDOW_LEN},
    decreasing_window_iter::DecreasingWindows,
    roaringish::MAX_VALUE,
    utils::{normalize, tokenize}, RoaringishPacked,
};

#[derive(Debug)]
pub enum CommonTokens {
    // List(GxHashSet<Box<str>>),
    FixedNum(u32),
    Percentage(f64),
}

#[derive(Debug)]
struct Batch<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    batch_id: u32,
    hllp_tokens: HyperLogLogPlus<Box<str>, gxhash::GxBuildHasher>,

    next_token_id: u32,
    token_to_token_id: GxHashMap<Box<str>, u32>,

    // this 2 containers are in sync
    token_id_to_roaringish_packed: Vec<RoaringishPacked>,
    token_id_to_token: Vec<Box<str>>,

    // this 3 containers are in sync
    doc_ids: Vec<u32>,
    documents: Vec<D>,
    tokenized_docs: Vec<Vec<u32>>,
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
            hllp_tokens: HyperLogLogPlus::new(18, gxhash::GxBuildHasher::default()).unwrap(),
            next_token_id: 0,
            token_to_token_id: GxHashMap::new(),
            token_id_to_roaringish_packed: Vec::new(),
            token_id_to_token: Vec::new(),
            doc_ids: Vec::new(),
            documents: Vec::new(),
            tokenized_docs: Vec::new(),
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
        self.tokenized_docs.clear();
    }

    fn push(&mut self, doc_id: u32, content: &str, doc: D, count_freq: impl FnMut(&str)) {
        let tokenized_doc = self.index_doc(content, doc_id, count_freq);
        self.doc_ids.push(doc_id);
        self.documents.push(doc);
        self.tokenized_docs.push(tokenized_doc);
    }

    fn get_token_id(
        token: &str,
        hllp_tokens: &mut HyperLogLogPlus<Box<str>, gxhash::GxBuildHasher>,
        token_to_token_id: &mut GxHashMap<Box<str>, u32>,
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
        mut count_freq: impl FnMut(&str),
    ) -> Vec<u32> {
        let mut tokenized_doc = Vec::new();
        let mut token_id_to_positions: FxHashMap<u32, Vec<u32>> = FxHashMap::new();
        let content = normalize(content);
        for (pos, token) in tokenize(&content).enumerate().take(MAX_VALUE as usize) {
            let token_id = Self::get_token_id(
                &token,
                &mut self.hllp_tokens,
                &mut self.token_to_token_id,
                &mut self.token_id_to_token,
                &mut self.token_id_to_roaringish_packed,
                &mut self.next_token_id,
            );

            count_freq(token);

            token_id_to_positions
                .entry(token_id)
                .or_default()
                .push(pos as u32);
            tokenized_doc.push(token_id);
        }

        for (token_id, positions) in token_id_to_positions.iter() {
            self.token_id_to_roaringish_packed[*token_id as usize].push(doc_id, positions);
        }
        tokenized_doc
    }

    fn flush(
        &mut self,
        db: &DB<D>,
        rwtxn: &mut RwTxn,
        common_tokens: &HashSet<Box<str>>,
        mmap_size: &mut usize,
    ) {
        if self.doc_ids.is_empty() {
            return;
        }

        self.merge_common_tokens(common_tokens);

        db.write_token_to_roaringish_packed(
            &self.token_to_token_id,
            &self.token_id_to_roaringish_packed,
            mmap_size,
            self.batch_id,
        );
        db.write_doc_id_to_document(rwtxn, &self.doc_ids, &self.documents);

        self.batch_id += 1;
        self.clear();
    }

    fn merge_common_tokens(&mut self, common_tokens: &HashSet<Box<str>>) {
        if common_tokens.is_empty() {
            return;
        }

        for (tokenized_doc, doc_id) in self.tokenized_docs.iter().zip(self.doc_ids.iter()) {
            let mut token_id_to_positions: FxHashMap<u32, Vec<u32>> = FxHashMap::new();
            let it = DecreasingWindows::new(tokenized_doc, MAX_WINDOW_LEN);
            for (pos, token_ids) in it.enumerate() {
                let token_id = token_ids[0];
                let token = &self.token_id_to_token[token_id as usize];
                let is_first_token_rare = !common_tokens.contains(token.as_str());
                // if is_first_token_rare {
                //     for i in 1..token_ids.len() {
                //         let token_id = token_ids[i];
                //         let token = &self.token_id_to_token[token_id as usize];
                //         let is_token_rare = !common_tokens.contains(token.as_str());
                //         if is_token_rare {
                //             break;
                //         }
                //         let token: String = token_ids[..i]
                //             .iter()
                //             .map(|token_id| self.token_id_to_token[*token_id as usize].as_ref())
                //             .intersperse(" ")
                //             .collect();
                //         let token_id = Self::get_token_id(
                //             &token,
                //             &mut self.token_to_token_id,
                //             &mut self.token_id_to_token,
                //             &mut self.token_id_to_roaringish_packed,
                //             &mut self.next_token_id,
                //         );
                //         token_id_to_positions
                //             .entry(token_id)
                //             .or_default()
                //             .push(pos as u32);
                //     }
                // } else {
                //     for i in 1..token_ids.len() {
                //         let token_id = token_ids[i];
                //         let token = &self.token_id_to_token[token_id as usize];
                //         let is_token_rare = !common_tokens.contains(token.as_str());
                //         let token: String = token_ids[..i]
                //             .iter()
                //             .map(|token_id| self.token_id_to_token[*token_id as usize].as_ref())
                //             .intersperse(" ")
                //             .collect();
                //         let token_id = Self::get_token_id(
                //             &token,
                //             &mut self.token_to_token_id,
                //             &mut self.token_id_to_token,
                //             &mut self.token_id_to_roaringish_packed,
                //             &mut self.next_token_id,
                //         );
                //         token_id_to_positions
                //             .entry(token_id)
                //             .or_default()
                //             .push(pos as u32);
                //         if is_token_rare {
                //             break;
                //         }
                //     }
                // }

                for i in 1..token_ids.len() {
                    let token_id = token_ids[i];
                    let token = &self.token_id_to_token[token_id as usize];
                    let is_token_rare = !common_tokens.contains(token.as_str());
                    if is_first_token_rare && is_token_rare {
                        break;
                    }
                    let token: String = token_ids[..i + 1]
                        .iter()
                        .map(|token_id| self.token_id_to_token[*token_id as usize].as_ref())
                        .intersperse(" ")
                        .collect();
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
                    if is_token_rare {
                        break;
                    }
                }
            }

            for (token_id, positions) in token_id_to_positions.iter() {
                self.token_id_to_roaringish_packed[*token_id as usize].push(*doc_id, positions);
            }
        }
    }
}

pub struct Indexer {
    batch_size: Option<u32>,
    common_tokens: Option<CommonTokens>,
}

impl Indexer {
    pub fn new(batch_size: Option<u32>, common_tokens: Option<CommonTokens>) -> Self {
        Self {
            batch_size,
            common_tokens
        }
    }

    fn generate_common_tokens<'a>(
        &self,
        token_to_freq: &'a GxHashMap<Box<str>, u32>,
    ) -> HashSet<Box<str>> {
        let Some(common_tokens) = &self.common_tokens else {
            return HashSet::new();
        };
        match common_tokens {
            // CommonTokens::List(tokens) => tokens,
            CommonTokens::FixedNum(max) => {
                let max = (*max as usize).min(token_to_freq.len());
                let mut token_to_freq: Vec<_> = token_to_freq.into_iter().collect();
                token_to_freq.sort_unstable_by_key(|(_, freq)| Reverse(*freq));
                token_to_freq[0..max]
                    .iter()
                    .map(|(token, _)| (*token).clone())
                    .collect()
            }
            CommonTokens::Percentage(p) => {
                let max = (token_to_freq.len() as f64 * *p) as usize;
                let mut token_to_freq: Vec<_> = token_to_freq.into_iter().collect();
                token_to_freq.sort_unstable_by_key(|(_, freq)| Reverse(*freq));
                token_to_freq[0..max]
                    .iter()
                    .map(|(token, _)| (*token).clone())
                    .collect()
            }
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
        let mut it = docs.into_iter();

        let mut token_to_freq = GxHashMap::new();
        let mut next_doc_id = 0;
        let mut mmap_size = 0;

        let mut b = std::time::Instant::now();
        while let Some((content, doc)) = it.next() {
            let doc_id = next_doc_id;
            next_doc_id += 1;

            batch.push(doc_id, content.as_ref(), doc, |token| {
                let (_, freq) = token_to_freq
                    .raw_entry_mut()
                    .from_key(token)
                    .or_insert_with(|| (token.to_owned().into_boxed_str(), 0));
                *freq += 1;
            });

            if next_doc_id % batch_size == 0 {
                break;
            }
        }

        let common_tokens = self.generate_common_tokens(&token_to_freq);
        drop(token_to_freq);

        batch.flush(&db, &mut rwtxn, &common_tokens, &mut mmap_size);
        println!("flushed batch in {}", b.elapsed().as_secs());

        b = std::time::Instant::now();
        for (content, doc) in it {
            let doc_id = next_doc_id;
            next_doc_id += 1;

            batch.push(doc_id, content.as_ref(), doc, |_| {});

            if next_doc_id % batch_size == 0 {
                batch.flush(&db, &mut rwtxn, &common_tokens, &mut mmap_size);
                println!("flushed batch in {}", b.elapsed().as_secs());
                b = std::time::Instant::now();
            }
        }

        batch.flush(&db, &mut rwtxn, &common_tokens, &mut mmap_size);

        let number_of_distinct_tokens = batch.estimate_number_of_distinct_tokens();

        db.write_common_tokens(&mut rwtxn, &common_tokens);
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
