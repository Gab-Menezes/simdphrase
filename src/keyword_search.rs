use std::cmp::Ordering;

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::FxHashMap;
use roaring::RoaringBitmap;

use crate::{db::DB, utils::{normalize, tokenize}};

pub trait KeywordSearch {
    fn search(&self, q: &str) -> RoaringBitmap;
}

impl KeywordSearch for DB {
    fn search(&self, q: &str) -> RoaringBitmap {
        let q = normalize(q);
        let tokens: Vec<_> = tokenize(&q).collect();

        if tokens.len() == 0 {
            return RoaringBitmap::new();
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
        let mut final_tokens_repr = Vec::with_capacity(final_tokens.len());
        for token in deduped_tokens.iter() {
            let Some(token_id) = self.get_token_id(&rotxn, token) else {
                return RoaringBitmap::new();
            };

            let pl = self.get_posting_list(&rotxn, token_id);

            token_to_token_id.insert(token.as_str(), token_id);
            token_id_to_posting_list.insert(token_id, pl);
            final_tokens_repr.push(token_id);
        }

        if token_id_to_posting_list.len() == 1 {
            let token_id = final_tokens_repr.first().unwrap();
            let pl = token_id_to_posting_list.get(token_id).unwrap();
            let mut token_id_to_positions = FxHashMap::new();

            let mut doc_ids = RoaringBitmap::new();
            for (doc_id, positions) in pl.doc_ids.iter().zip(pl.positions.iter()) {
                token_id_to_positions.insert(*token_id, positions);

                if Self::begin_phrase_search(
                    &final_tokens,
                    &token_to_token_id,
                    &token_id_to_positions,
                    query_len,
                ) {
                    doc_ids.push(doc_id);
                }
            }

            return doc_ids;
        }

        let mut temp: Vec<_> = token_id_to_posting_list
            .iter()
            .map(|(token_id, pl)| (*token_id, pl.doc_ids.iter().peekable(), &pl.positions))
            .collect();
        temp.sort_unstable_by_key(|(_, it, _)| it.len());

        // let's do a vec instead of a hashmap, since the number of total
        // tokens is probably low
        let mut token_ids = Vec::with_capacity(temp.len());
        let mut its = Vec::with_capacity(temp.len());
        let mut positionss = Vec::with_capacity(temp.len());
        let mut token_id_to_positions = FxHashMap::with_capacity(temp.len());
        for (token_id, it, positions) in temp {
            token_ids.push(token_id);
            its.push(it);
            positionss.push(positions);
        }

        let mut doc_ids = RoaringBitmap::new();

        'outer: loop {
            let Some(current_doc_id) = its.first_mut().unwrap().next() else {
                break;
            };

            let mut max = current_doc_id;
            let mut i = 0;
            'inner: for (j, it) in its.iter_mut().enumerate().skip(1) {
                loop {
                    let Some(doc_id) = it.peek() else {
                        break 'outer;
                    };

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
                            max = *doc_id;
                            i = j;
                            break 'inner;
                        }
                    }
                }
            }

            if current_doc_id == max {
                for (token_id, (it, positions)) in
                    token_ids.iter().zip(its.iter().zip(positionss.iter()))
                {
                    let idx = positions.len() - it.len() - 1;
                    token_id_to_positions.insert(*token_id, &positions[idx]);
                }

                if Self::begin_phrase_search(
                    &final_tokens,
                    &token_to_token_id,
                    &token_id_to_positions,
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

                        if *doc_id < max {
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