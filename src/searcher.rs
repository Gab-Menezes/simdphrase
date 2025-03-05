use std::{collections::HashSet, path::Path};

use crate::{DB, DbError, Intersection, SearchError, Stats, db::Document, error::GetDocumentError};
use memmap2::Mmap;
use rkyv::{Archive, Deserialize, de::Pool, rancor::Strategy};

/// Final result of a search operation.
pub struct SearchResult<'a, D: Document>(pub Result<Vec<u32>, SearchError>, &'a Searcher<D>);
impl<D: Document> SearchResult<'_, D> {
    /// Number of documents that matched the search query.
    pub fn len(&self) -> Option<usize> {
        self.0.as_ref().map(|p| p.len()).ok()
    }

    /// Returns the internal document IDs that matched the search query.
    pub fn get_internal_document_ids(&self) -> Option<&[u32]> {
        self.0.as_ref().map(|p| p.as_slice()).ok()
    }

    /// Gets the archived version of the documents that matched the search query.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use [Self::get_documents] instead.
    pub fn get_archived_documents(
        &self,
        cb: impl FnOnce(Vec<&D::Archived>),
    ) -> Result<(), GetDocumentError> {
        let Some(doc_ids) = self.get_internal_document_ids() else {
            return Ok(());
        };

        self.1.get_archived_documents(doc_ids, cb)
    }

    /// Gets the deserialized version of the documents that matched the search query.
    pub fn get_documents(&self) -> Result<Vec<D>, GetDocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        let Some(doc_ids) = self.get_internal_document_ids() else {
            return Ok(Vec::new());
        };

        self.1.get_documents(doc_ids)
    }
}

/// Object responsible for searching the database.
pub struct Searcher<D: Document> {
    db: DB<D>,
    common_tokens: HashSet<Box<str>>,
    mmap: Mmap,
}

impl<D: Document> Searcher<D> {
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
    pub fn search<I: Intersection>(&self, q: &str) -> SearchResult<D> {
        let stats = Stats::default();
        self.search_with_stats::<I>(q, &stats)
    }

    /// Searches by the query `q`, allowing the user to pass a [Stats] object.
    pub fn search_with_stats<I: Intersection>(&self, q: &str, stats: &Stats) -> SearchResult<D> {
        SearchResult(
            self.db
                .search::<I>(q, stats, &self.common_tokens, &self.mmap),
            self,
        )
    }

    /// Gets the archived version of the documents.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use [Self::get_documents] instead.
    pub fn get_archived_documents(
        &self,
        doc_ids: &[u32],
        cb: impl FnOnce(Vec<&D::Archived>),
    ) -> Result<(), GetDocumentError> {
        self.db.get_archived_documents(doc_ids, cb)
    }

    /// Gets the archived version of a documents.
    ///
    /// This avoids having to deserialize, but it's necessary to use a callback
    /// due to the lifetime of the transaction.
    ///
    /// If you want the documents deserialized, use [Self::get_document] instead.
    pub fn get_archived_document(
        &self,
        doc_id: u32,
        cb: impl FnOnce(&D::Archived),
    ) -> Result<(), GetDocumentError> {
        self.db.get_archived_document(doc_id, cb)
    }

    /// Gets the deserialized version of the documents.
    pub fn get_documents(&self, doc_ids: &[u32]) -> Result<Vec<D>, GetDocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        self.db.get_documents(doc_ids)
    }

    /// Gets the deserialized version of a documents.
    pub fn get_document(&self, doc_id: u32) -> Result<D, GetDocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        self.db.get_document(doc_id)
    }
}
