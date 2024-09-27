#![feature(hash_raw_entry)]
#![feature(array_windows)]
#![feature(iter_intersperse)]
#![feature(new_uninit)]
#![feature(hint_assert_unchecked)]
#![feature(core_intrinsics)]
#![feature(debug_closure_helpers)]
#![feature(maybe_uninit_fill)]
#![feature(maybe_uninit_write_slice)]
#![feature(vec_push_within_capacity)]
#![feature(trivial_bounds)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(maybe_uninit_uninit_array)]

mod indexer;
mod db;
mod utils;
mod codecs;
mod roaringish;
mod searcher;

pub use db::{DB, Stats};
pub use indexer::Indexer;
pub use indexer::CommonTokens;
pub use searcher::Searcher;
pub use utils::{normalize, tokenize};
pub use roaringish::RoaringishPacked;
pub use roaringish::BorrowRoaringishPacked;
pub use roaringish::intersect::Intersect;
pub use roaringish::intersect::naive;
pub use roaringish::intersect::simd;