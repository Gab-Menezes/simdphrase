use crate::roaringish::Roaringish;

#[derive(Debug, Default)]
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

    pub fn push_unchecked(&mut self, doc_id: u32, positions: Roaringish) {
        self.len_sum += positions.inner.len() as u64;
        self.doc_ids.push(doc_id);
        self.positions.push(positions);
    }
}
