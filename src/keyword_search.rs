use std::{cell::UnsafeCell, cmp::Ordering, iter::Peekable, slice::Iter};

use ahash::{AHashMap, AHashSet, HashMapExt};
use fxhash::FxHashMap;
use rkyv::vec::ArchivedVec;
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

        let mut its = Vec::with_capacity(final_tokens.len());
        let mut positionss = Vec::with_capacity(final_tokens.len());
        for t in final_tokens.iter() {
            let token_id = token_to_token_id.get(t.as_str()).unwrap();
            let pl = token_id_to_posting_list.get(token_id).unwrap();

            its.push(pl.doc_ids.iter().peekable());
            positionss.push(&pl.positions);
        }

        unsafe { 
            std::hint::assert_unchecked(its.len() == positionss.len());
            std::hint::assert_unchecked(its.len() >= 1);
        };
    

        intersect(query_len, its, positionss)
    }
}

#[inline(always)]
fn intersect(query_len: usize, mut its: Vec<Peekable<Iter<u32>>>, positionss: Vec<&ArchivedVec<ArchivedVec<u32>>>) -> Vec<u32> {
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
            if begin_phrase_search(
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

#[inline(always)]
fn begin_phrase_search(
    its: &[Peekable<Iter<u32>>],
    positionss: &[&ArchivedVec<ArchivedVec<u32>>],
    query_len: usize,
) -> bool {
    unsafe { 
        std::hint::assert_unchecked(its.len() == positionss.len());
        std::hint::assert_unchecked(its.len() >= 1);
    };

    let mut zipped_it = its.iter().zip(positionss);

    let (it, positions) = zipped_it.next().unwrap();
    let idx = positions.len() - it.len() - 1;
    let positions = unsafe { positions.get_unchecked(idx).as_ref() };
    let u = unsafe { *positions.first().unwrap_unchecked() };
    let mut v = u;

    for (it, positions) in zipped_it {
        let idx = positions.len() - it.len() - 1;
        let positions = unsafe { positions.get_unchecked(idx).as_ref() };
        let idx = match positions.binary_search(&v) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        let Some(next) = positions.get(idx) else {
            return false;
        };
        v = *next
    }

    if (v - u) as usize == query_len - 1 {
        true
    } else {
        phrase_search(
            its,
            positionss,
            v - query_len as u32,
            query_len,
        )
    }
}

#[inline(always)]
fn phrase_search<'a, 'b: 'a>(
    its: &[Peekable<Iter<u32>>],
    positionss: &[&ArchivedVec<ArchivedVec<u32>>],
    mut position: u32,
    query_len: usize,
) -> bool {
    loop {
        let mut zipped_it = its.iter().zip(positionss);

        let (it, positions) = zipped_it.next().unwrap();
        let idx = positions.len() - it.len() - 1;
        let positions = unsafe { positions.get_unchecked(idx).as_ref() };

        let idx = match positions.binary_search(&position) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        let Some(next) = positions.get(idx) else {
            return false;
        };
        let u = *next;
        let mut v = u;

        for (it, positions) in zipped_it {
            let idx = positions.len() - it.len() - 1;
            let positions = unsafe { positions.get_unchecked(idx).as_ref() };
            let idx = match positions.binary_search(&v) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            };
            let Some(next) = positions.get(idx) else {
                return false;
            };
            v = *next
        }

        if (v - u) as usize == query_len - 1 {
            return true;
        }

        position = v - query_len as u32;
    }
}
