use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Archive)]
#[archive_attr(derive(Debug))]
pub struct PostingList {
    pub(crate) doc_ids: Vec<u32>,
    pub(crate) positions: Vec<Vec<u32>>,
}

impl PostingList {
    pub fn new(doc_ids: Vec<u32>, positions: Vec<Vec<u32>>) -> Self {
        Self { doc_ids, positions }
    }
}
