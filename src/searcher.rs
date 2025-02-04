use std::{collections::HashSet, path::Path};

use crate::{
    db::Document, error::DocumentError, DatabaseError, Db, Intersection, SearchError, Stats,
};
use memmap2::Mmap;
use rkyv::{de::Pool, rancor::Strategy, Archive, Deserialize};

/// Final result of a search operation.
pub struct SearchResult<'a, D: Document>(pub Result<Vec<u32>, SearchError>, &'a Searcher<D>);
impl<D: Document> SearchResult<'_, D> {
    /// Number of documents that matched the search query.
    pub fn len(&self) -> Option<usize> {
        self.0.as_ref().map(|p| p.len()).ok()
    }

    /// Returns the internal document IDs that matched the search query.
    pub fn internal_document_ids(&self) -> Option<&[u32]> {
        self.0.as_ref().map(|p| p.as_slice()).ok()
    }

    /// Returns the archived version of the documents that matched the search
    /// query.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use
    /// [`documents()`](Self::documents) instead.
    pub fn archived_documents(
        &self,
        cb: impl FnOnce(Vec<&D::Archived>),
    ) -> Result<(), DocumentError> {
        let Some(doc_ids) = self.internal_document_ids() else {
            return Ok(());
        };

        self.1.archived_documents(doc_ids, cb)
    }

    /// Returns the deserialized version of the documents that matched the
    /// search query.
    pub fn documents(&self) -> Result<Vec<D>, DocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        let Some(doc_ids) = self.internal_document_ids() else {
            return Ok(Vec::new());
        };

        self.1.documents(doc_ids)
    }
}

/// Object responsible for searching the database.
pub struct Searcher<D: Document> {
    db: Db<D>,
    common_tokens: HashSet<Box<str>>,
    mmap: Mmap,
}

impl<D: Document> Searcher<D> {
    /// Create a new searcher object.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, DatabaseError> {
        let (db, common_tokens, mmap) = Db::open(path)?;
        Ok(Self {
            db,
            common_tokens,
            mmap,
        })
    }

    /// Searches by `query`.
    pub fn search<I: Intersection>(&self, query: &str) -> SearchResult<D> {
        let stats = Stats::default();
        self.search_with_stats::<I>(query, &stats)
    }

    /// Searches by the query `q`, allowing the user to pass a [`Stats`] object.
    pub fn search_with_stats<I: Intersection>(
        &self,
        query: &str,
        stats: &Stats,
    ) -> SearchResult<D> {
        SearchResult(
            self.db
                .search::<I>(query, stats, &self.common_tokens, &self.mmap),
            self,
        )
    }

    /// Returns the archived version of the documents.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use
    /// [`documents()`](Self::documents) instead.
    pub fn archived_documents(
        &self,
        document_ids: &[u32],
        cb: impl FnOnce(Vec<&D::Archived>),
    ) -> Result<(), DocumentError> {
        self.db.archived_documents(document_ids, cb)
    }

    /// Returns the archived version of a documents.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use
    /// [`document()`](Self::document) instead.
    pub fn archived_document(
        &self,
        document_id: u32,
        cb: impl FnOnce(&D::Archived),
    ) -> Result<(), DocumentError> {
        self.db.archived_document(document_id, cb)
    }

    /// Returns the deserialized version of the documents.
    pub fn documents(&self, document_ids: &[u32]) -> Result<Vec<D>, DocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        self.db.documents(document_ids)
    }

    /// Returns the deserialized version of a documents.
    pub fn document(&self, document_id: u32) -> Result<D, DocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        self.db.document(document_id)
    }
}
