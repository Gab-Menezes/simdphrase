#![feature(hash_raw_entry)]
#![feature(array_windows)]
#![feature(iter_intersperse)]

mod indexer;
mod db;
mod document;
mod pl;
mod utils;
mod codecs;
mod keyword_search;

pub use db::DB;
pub use db::ShardsInfo;
pub use keyword_search::KeywordSearch;
pub use indexer::Indexer;