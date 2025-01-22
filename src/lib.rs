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

#[cfg(target_feature = "avx512f")]
pub use roaringish::intersect::simd::SimdIntersect;
pub use roaringish::intersect::Intersection;
pub use searcher::{SearchResult, Searcher};
