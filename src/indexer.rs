use std::{cell::Cell, cmp::Reverse, collections::HashSet, path::Path};

use ahash::{AHashMap, AHashSet, HashMapExt, HashSetExt};
use fxhash::{FxHashMap, FxHashSet};
use heed::RwTxn;
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};

use crate::{
    db::DB,
    roaringish::{RoaringishPacked, MAX_VALUE},
    utils::{normalize, tokenize, MAX_SEQ_LEN},
};

#[derive(Debug)]
pub enum CommonTokens {
    // List(AHashSet<Box<str>>),
    FixedNum(u32),
    Percentage(f64),
}

#[derive(Debug)]
struct Batch<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    token_to_packed: AHashMap<Box<str>, RoaringishPacked>,

    // this 3 containers are in sync
    doc_ids: Vec<u32>,
    documents: Vec<D>,
    tokenized_docs: Vec<Vec<Box<str>>>,
}

impl<D> Batch<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    fn new(batch_size: Option<u32>, avg_document_tokens: Option<u32>) -> Self {
        match batch_size {
            Some(batch_size) => match avg_document_tokens {
                Some(avg_document_tokens) => {
                    // Heap's Law
                    let len = (1.705 * (batch_size as f64 * avg_document_tokens as f64).powf(0.786))
                        .ceil() as usize;
                    Self {
                        token_to_packed: AHashMap::with_capacity(len),
                        tokenized_docs: Vec::with_capacity(len),
                        doc_ids: Vec::with_capacity(batch_size as usize),
                        documents: Vec::with_capacity(batch_size as usize),
                    }
                }
                None => Self {
                    token_to_packed: AHashMap::new(),
                    tokenized_docs: Vec::new(),
                    doc_ids: Vec::with_capacity(batch_size as usize),
                    documents: Vec::with_capacity(batch_size as usize),
                },
            },
            None => Self {
                token_to_packed: AHashMap::new(),
                tokenized_docs: Vec::new(),
                doc_ids: Vec::new(),
                documents: Vec::new(),
            },
        }
    }

    fn clear(&mut self) {
        self.token_to_packed.clear();
        self.doc_ids.clear();
        self.documents.clear();
        self.tokenized_docs.clear();
    }

    fn push(&mut self, doc_id: u32, content: &str, doc: D, count_freq: impl FnMut(&Box<str>)) {
        let tokenized_doc = self.index_doc(content, doc_id, count_freq);
        self.doc_ids.push(doc_id);
        self.documents.push(doc);
        self.tokenized_docs.push(tokenized_doc);
    }

    fn analyze_common_tokens_sequence(
        sequence: &[&str],
        begin_pos: usize,
        token_to_positions: &mut AHashMap<Box<str>, Vec<u32>>,
    ) {
        if sequence.len() <= 1 {
            return;
        }
        for i in 0..(sequence.len() - 1) {
            let b = i + 2;
            let e = (sequence.len() + 1).min(i + MAX_SEQ_LEN + 1);
            for j in b..e {
                let token: String = sequence[i..j]
                    .into_iter()
                    .map(|s| *s)
                    .intersperse(" ")
                    .collect();
                let (_, pos) = token_to_positions
                    .raw_entry_mut()
                    .from_key(token.as_str())
                    .or_insert_with(|| (token.into_boxed_str(), Vec::new()));
                pos.push((begin_pos + i) as u32);
            }
        }
    }

    fn index_doc(
        &mut self,
        content: &str,
        doc_id: u32,
        mut count_freq: impl FnMut(&Box<str>),
    ) -> Vec<Box<str>> {
        let mut tokenized_doc = Vec::new();
        let mut token_to_positions: AHashMap<&str, Vec<u32>> = AHashMap::new();
        let content = normalize(content);
        for (pos, token) in tokenize(&content).enumerate().take(MAX_VALUE as usize) {
            let owned_token = token.to_owned().into_boxed_str();
            count_freq(&owned_token);

            token_to_positions
                .entry(token)
                .or_default()
                .push(pos as u32);
            tokenized_doc.push(owned_token);
        }

        for (token, positions) in token_to_positions.iter() {
            let (_, packed) = self
                .token_to_packed
                .raw_entry_mut()
                .from_key(*token)
                .or_insert_with(|| {
                    (
                        token.to_string().into_boxed_str(),
                        RoaringishPacked::default(),
                    )
                });
            packed.push(doc_id, positions);
        }
        tokenized_doc
    }

    fn flush(&mut self, db: &DB<D>, rwtxn: &mut RwTxn, common_tokens: &HashSet<Box<str>>, mmap_size: &mut usize) {
        if self.doc_ids.is_empty() {
            return;
        }

        self.generate_common_tokens(common_tokens);

        db.write_token_to_roaringish_packed(rwtxn, &self.token_to_packed, mmap_size);
        db.write_doc_id_to_document(rwtxn, &self.doc_ids, &self.documents);

        self.clear();
    }

    fn generate_common_tokens(&mut self, common_tokens: &HashSet<Box<str>>) {
        let mut sequence = Vec::new();
        for (token_id_repr, doc_id) in self.tokenized_docs.iter().zip(self.doc_ids.iter()) {
            let mut token_to_positions: AHashMap<Box<str>, Vec<u32>> = AHashMap::new();
            sequence.clear();
            let mut begin_pos = 0;
            for (pos, token) in token_id_repr.iter().enumerate() {
                if common_tokens.contains(token.as_str()) {
                    sequence.push(token.as_str());
                    continue;
                }

                if sequence.len() <= 1 {
                    sequence.clear();
                    begin_pos = pos + 1;
                    continue;
                }

                Self::analyze_common_tokens_sequence(&sequence, begin_pos, &mut token_to_positions);

                sequence.clear();
                begin_pos = pos + 1;
            }

            Self::analyze_common_tokens_sequence(&sequence, begin_pos, &mut token_to_positions);

            for (token, positions) in token_to_positions.iter() {
                let (_, packed) = self
                    .token_to_packed
                    .raw_entry_mut()
                    .from_key(token)
                    .or_insert_with(|| {
                        (
                            token.to_string().into_boxed_str(),
                            RoaringishPacked::default(),
                        )
                    });
                packed.push(*doc_id, positions);
            }
        }
    }
}

pub struct Indexer {
    batch_size: Option<u32>,
    common_tokens: Option<CommonTokens>,
    avg_document_tokens: Option<u32>,
}

impl Indexer {
    pub fn new(
        batch_size: Option<u32>,
        common_tokens: Option<CommonTokens>,
        avg_document_tokens: Option<u32>,
    ) -> Self {
        Self {
            batch_size,
            common_tokens,
            avg_document_tokens,
        }
    }

    fn generate_common_tokens<'a>(
        &self,
        token_to_freq: &'a AHashMap<Box<str>, u32>,
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

        let mut batch = Batch::new(self.batch_size, self.avg_document_tokens);

        let batch_size = self.batch_size.unwrap_or(u32::MAX);
        let mut it = docs.into_iter();

        let mut token_to_freq = AHashMap::new();
        let mut next_doc_id = 0;
        let mut mmap_size = 0;
        // compute the first batch to get common tokens
        let mut b = std::time::Instant::now();
        while let Some((content, doc)) = it.next() {
            let doc_id = next_doc_id;
            next_doc_id += 1;

            batch.push(doc_id, content.as_ref(), doc, |owned_token| {
                let (_, freq) = token_to_freq
                    .raw_entry_mut()
                    .from_key(owned_token.as_str())
                    .or_insert_with(|| (owned_token.clone(), 0));
                *freq += 1;
            });

            if next_doc_id % batch_size == 0 {
                break;
            }
        }

        let common_tokens = self.generate_common_tokens(&token_to_freq);

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
        println!("flushed batch in {}", b.elapsed().as_secs());

        b = std::time::Instant::now();
        db.write_common_tokens(&mut rwtxn, &common_tokens);
        db.generate_mmap_file(mmap_size, &mut rwtxn);
        println!("write mmap {}", b.elapsed().as_secs());
        rwtxn.commit().unwrap();


        next_doc_id
    }
}
