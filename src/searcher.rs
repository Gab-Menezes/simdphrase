use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use memmap2::Mmap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};

use crate::{roaringish::intersect::Intersect, Stats, DB};

pub struct Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    db: DB<D>,
    mmap: Mmap,
}

impl<D> Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    pub fn new(path: &Path, db_size: usize) -> Option<Self> {
        let paths: Option<Vec<PathBuf>> = std::fs::read_dir(path)
            .ok()?
            .map(|path| path.ok().map(|p| p.path()))
            .collect();
        let mut paths = paths?;
        paths.sort_unstable();

        let (db, mmap) = DB::open(path, db_size);
        Some(Self { db, mmap })
    }

    pub fn search<I: Intersect>(&self, q: &str, stats: &Stats) -> Vec<u32> {
        self.db.search::<I>(q, stats, &self.mmap)
    }

    // pub fn foo(&self, q: &str)
    // where
    //     D: Display
    // {
    //     self.shards
    //         .iter()
    //         .for_each(|shard| {
    //             println!("New shard");
    //             shard.get_documents_by_ids(shard.search(q))
    //         });
    // }
}