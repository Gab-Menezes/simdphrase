use rkyv::{Archive, Deserialize, Serialize};

use crate::Roaringish;

#[derive(Debug, Default, Serialize, Deserialize, Archive)]
#[archive_attr(derive(Debug))]
pub struct PostingList {
    pub(crate) doc_ids: Vec<u32>,
    pub(crate) positions: Vec<Roaringish>,
    pub(crate) len_sum: u64
}

impl PostingList {
    pub fn new(doc_ids: Vec<u32>, positions: Vec<Roaringish>) -> Self {
        let len_sum = positions
        .iter()
        .map(|r| r.inner.len() as u64)
        .sum();

        Self { doc_ids, positions, len_sum }
    }
}
