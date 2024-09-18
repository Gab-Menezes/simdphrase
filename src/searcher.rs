use std::{
    fmt::{Debug, Display},
    path::{Path, PathBuf},
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Deserialize,
    Serialize,
};

use crate::{roaringish::intersect::Intersect, Stats, DB};

pub struct Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive,
{
    pub(crate) shards: Box<[DB<D>]>,
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

        let shards = paths.iter().map(|path| DB::open(path, db_size)).collect();
        Some(Self { shards })
    }

    pub fn search<I: Intersect>(&self, q: &str, stats: &Stats) -> Vec<u32> {
        self.shards
            .iter()
            .map(|shard| shard.search::<I>(q, stats))
            .flatten()
            .collect()
    }

    pub fn get_shard(&self, idx: usize) -> Option<&DB<D>> {
        self.shards.get(idx)
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

impl<D> Searcher<D>
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Send
        + Sync
        + 'static,
{
    pub fn par_search<I: Intersect>(&self, q: &str, stats: &Stats) -> Vec<u32> {
        self.shards
            .par_iter()
            .map(|shard| shard.search::<I>(q, stats))
            .flatten()
            .collect()
    }
}
