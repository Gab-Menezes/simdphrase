#![feature(hash_raw_entry)]
#![feature(array_windows)]
#![feature(iter_intersperse)]
#![feature(debug_closure_helpers)]
#![feature(vec_push_within_capacity)]
#![feature(trivial_bounds)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]

//! Extremely fast phrase search implementation.
//!
//! ## Overview
//!
//! This implementation follows some of the ideas proposed in this
//! [blog post](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm)
//! by [Doug Turnbull](https://softwaredoug.com/). The full explanation on how the internals
//! work can be found in [here](https://gab-menezes.github.io/2025/01/13/using-the-most-unhinged-avx-512-instruction-to-make-the-fastest-phrase-search-algo.html).
//!
//! This crate uses the [log] crate for logging during indexing.
//!
//! It's highly recommended to compile this crate with `-C llvm-args=-align-all-functions=6`.
//!
//! ## Usage
//!
//! ```rust
//! use phrase_search::{CommonTokens, Indexer, SimdIntersect};
//!
//! // Creates a new indexer that can be reused, it will index 300_000 documents
//! // in each batch and will use the top 50 most common tokens to speed up the search,
//! // by merging them.
//! let indexer = Indexer::new(Some(300_000), Some(CommonTokens::FixedNum(50)));
//!
//! let docs = vec![
//!     ("look at my beautiful cat", 0),
//!     ("this is a document", 50),
//!     ("look at my dog", 25),
//!     ("look at my beautiful hamster", 35),
//! ];
//! let index_name = "./index";
//! let db_size = 1024 * 1024;
//!
//! // Indexes the documents returned by the iterator `it`.
//! // The index will be created at `index_name` with the given `db_size`.
//! let (searcher, num_indexed_documents) = indexer.index(docs, index_name, db_size)?;
//!
//! // Search by the string "78"
//! let result = searcher.search::<SimdIntersect>("at my beautiful")?;
//! // This should return `[0, 35]`
//! let documents = result.get_documents()?;
//! ```

mod allocator;
mod codecs;
mod db;
mod decreasing_window_iter;
mod error;
mod indexer;
mod roaringish;
mod searcher;
mod stats;
mod utils;

use allocator::Aligned64;
use db::DB;
use roaringish::BorrowRoaringishPacked;
use roaringish::RoaringishPacked;
use utils::{normalize, tokenize};

pub use db::Document;
pub use error::{DbError, GetDocumentError, SearchError};
pub use indexer::CommonTokens;
pub use indexer::Indexer;
pub use stats::Stats;

pub use roaringish::intersect::naive::NaiveIntersect;

pub use roaringish::intersect::Intersection;
#[cfg(target_feature = "avx512f")]
pub use roaringish::intersect::simd::SimdIntersect;
pub use searcher::{SearchResult, Searcher};
