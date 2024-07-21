use std::borrow::Cow;

use heed::{byteorder::BigEndian, types::U64, BoxedError};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Some ideias for the PL
///
/// * Maybe make the positions to save memory
///     enum Positions {
///         Bitmap(Vec<RoaringBitmap>)
///         Vec(Vec<Vec<u32>>)
///     }
/// and we decoding the Vec variant we could
/// construct a Box<[&[u32]]> to avoid copying
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct PostingList {
    pub(crate) doc_ids: Vec<u32>,
    pub(crate) positions: Vec<Vec<u32>>,
}

impl PostingList {
    pub fn new(doc_ids: Vec<u32>, positions: Vec<Vec<u32>>) -> Self {
        Self {
            doc_ids,
            positions
        }
    }
}

pub struct UnsafePostingList<'a> {
    pub(crate) doc_ids: &'a [u32],
    pub(crate) positions: Box<[&'a [u32]]>,
}