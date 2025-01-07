#![feature(hash_raw_entry)]
#![feature(array_windows)]
#![feature(iter_intersperse)]
#![feature(debug_closure_helpers)]
#![feature(maybe_uninit_fill)]
#![feature(maybe_uninit_write_slice)]
#![feature(vec_push_within_capacity)]
#![feature(trivial_bounds)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(maybe_uninit_uninit_array)]
#![feature(allocator_api)]
#![feature(str_as_str)]
#![feature(pointer_is_aligned_to)]
#![feature(array_chunks)]
#![feature(maybe_uninit_slice)]
#![feature(new_range_api)]

mod allocator;
mod codecs;
mod db;
mod decreasing_window_iter;
mod indexer;
mod roaringish;
mod searcher;
mod utils;

pub use db::{Stats, DB};
pub use indexer::CommonTokens;
pub use indexer::Indexer;

pub use roaringish::intersect::binary_search;
pub use roaringish::intersect::naive;

// #[cfg(all(
//     target_feature = "avx512f",
//     target_feature = "avx512bw",
//     target_feature = "avx512vl",
//     target_feature = "avx512vbmi2",
//     target_feature = "avx512dq",
// ))]
pub use roaringish::intersect::simd;

pub use allocator::Aligned64;
pub use roaringish::intersect::Intersect;
pub use roaringish::BorrowRoaringishPacked;
pub use roaringish::RoaringishPacked;
pub use searcher::Searcher;
pub use utils::{normalize, tokenize};
