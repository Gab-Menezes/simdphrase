use std::{collections::HashSet, path::Path};

use memmap2::Mmap;
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};

use crate::{
    DbError, Intersection, SearchError, Stats, DB
};

/// Final result of a search operation.
pub struct SearchResult(pub Result<Vec<u32>, SearchError>);
impl SearchResult {
    /// Number of documents that matched the search query.
    pub fn len(&self) -> usize {
        self.doc_ids().len()
    }

    /// Returns the internal document IDs that matched the search query.
    pub fn doc_ids(&self) -> &[u32] {
        self.0.as_ref().map(|p| p.as_slice()).unwrap_or_default()
    }
}

/// Object responsible for searching the database.
pub struct Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    db: DB<D>,
    common_tokens: HashSet<Box<str>>,
    mmap: Mmap,
}

impl<D> Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    /// Create a new searcher object.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, DbError> {
        let (db, common_tokens, mmap) = DB::open(path)?;
        Ok(Self {
            db,
            common_tokens,
            mmap,
        })
    }

    /// Searches by the query `q`
    pub fn search<I: Intersection>(&self, q: &str) -> SearchResult {
        let stats = Stats::default();
        self.search_with_stats::<I>(q, &stats)
    }

    /// Searches by the query `q`, allowing the user to pass a [Stats] object.
    pub fn search_with_stats<I: Intersection>(&self, q: &str, stats: &Stats) -> SearchResult {
        SearchResult(
            self.db
                .search::<I>(q, stats, &self.common_tokens, &self.mmap),
        )
    }
}
