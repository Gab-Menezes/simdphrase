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

mod indexer;
mod db;
mod document;
mod pl;
mod utils;
mod codecs;
mod keyword_search;
mod roaringish;

pub use db::DB;
pub use db::ShardsInfo;
pub use keyword_search::KeywordSearch;
pub use indexer::Indexer;
pub use roaringish::Roaringish;