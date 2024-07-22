use std::{cell::UnsafeCell, cmp::Ordering};

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::FxHashMap;
use roaring::RoaringBitmap;

use crate::{
    db::DB,
    utils::{normalize, tokenize},
};

pub trait KeywordSearch {
    fn search(&self, q: &str) -> Vec<u32>;
}

impl KeywordSearch for DB {
    fn search(&self, q: &str) -> Vec<u32> {
        let q = normalize(q);
        let tokens: Vec<_> = tokenize(&q).collect();

        if tokens.len() == 0 {
            return Vec::new();
        }

        let rotxn = self.env.read_txn().unwrap();
        if tokens.len() == 1 {
            return self.single_token_search(&rotxn, &tokens);
        }

        let final_tokens = self.merge_tokens(&tokens);

        if final_tokens.len() == 1 {
            return self.single_token_search(&rotxn, &final_tokens);
        }

        let query_len = tokens.len();

        let deduped_tokens: AHashSet<_> = final_tokens.iter().collect();
        let mut token_to_token_id = AHashMap::with_capacity(deduped_tokens.len());
        let mut token_id_to_posting_list = FxHashMap::with_capacity(deduped_tokens.len());
        for token in deduped_tokens.iter() {
            let Some(token_id) = self.get_token_id(&rotxn, token) else {
                return Vec::new();
            };

            let pl = self.get_posting_list(&rotxn, token_id);

            token_to_token_id.insert(token.as_str(), token_id);
            token_id_to_posting_list.insert(token_id, pl);
        }


        // let's do a vec instead of a hashmap, since the number of total
        // tokens is probably low
        let mut token_ids = Vec::with_capacity(final_tokens.len());
        let mut its = Vec::with_capacity(final_tokens.len());
        let mut positionss = Vec::with_capacity(final_tokens.len());
        for t in final_tokens.iter() {
            let token_id = token_to_token_id.get(t.as_str()).unwrap();
            let pl = token_id_to_posting_list.get(token_id).unwrap();

            token_ids.push(token_id);
            its.push(pl.doc_ids.iter().peekable());
            positionss.push(&pl.positions);
        }

        let mut doc_ids = Vec::new();

        'outer: loop {
            let Some(current_doc_id) = its.first_mut().unwrap().next() else {
                break;
            };

            let current_doc_id = *current_doc_id;
            let mut max = current_doc_id;
            let mut i = 0;

            'inner: for (j, it) in its.iter_mut().enumerate().skip(1) {
                loop {
                    let Some(doc_id) = it.peek() else {
                        break 'outer;
                    };
                    let doc_id = **doc_id;

                    match doc_id.cmp(&max) {
                        Ordering::Less => {
                            it.next();
                            continue;
                        }
                        Ordering::Equal => {
                            it.next();
                            continue 'inner;
                        }
                        Ordering::Greater => {
                            max = doc_id;
                            i = j;
                            break 'inner;
                        }
                    }
                }
            }

            if current_doc_id == max {
                if Self::begin_phrase_search(
                    &its,
                    &positionss,
                    query_len,
                ) {
                    doc_ids.push(current_doc_id);
                }
            } else {
                for (j, it) in its.iter_mut().enumerate() {
                    if i == j {
                        continue;
                    }

                    loop {
                        let Some(doc_id) = it.peek() else {
                            break 'outer;
                        };

                        if **doc_id < max {
                            it.next();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        doc_ids
    }
}
