#![feature(new_uninit)]

use bincode::{
    config::{BigEndian, WithOtherEndian},
    DefaultOptions, Options,
};
use clap::{Args, Parser, Subcommand};
use phrase_search::{KeywordSearch, ShardsInfo, DB};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator},
    slice::ParallelSlice,
};
use serde::{Deserialize, Serialize};
use std::{borrow::{Borrow, Cow}, cmp::Ordering, collections::BTreeSet, fs::OpenOptions, io::Write, iter::{Enumerate, Peekable}, path::PathBuf, process::{Command, Stdio}, str::FromStr, sync::{atomic::{AtomicU32, Ordering::Relaxed}, RwLock}};

use ahash::{AHashSet, HashSet};
use roaring::RoaringBitmap;

#[derive(Parser, Debug)]
struct CommandArgs {
    #[command(subcommand)]
    ty: Ty,
}

#[derive(Subcommand, Debug)]
enum Ty {
    IndexFolder(IndexFolder),
    Search(Search),
}

#[derive(Args, Debug)]
struct IndexFolder {
    folder: PathBuf,
    index_name: PathBuf,
    files_per_index: usize,
    db_size: usize,
    max_doc_id: u32,

    #[arg(short, long)]
    recursive: bool,
}

#[derive(Args, Debug)]
struct Search {
    runs: u32,
    queries: PathBuf,
    index_name: PathBuf,
    db_size: usize,
}

fn index_folder(args: IndexFolder) {
    println!("Start Indexing");

    let info = DB::get_shards_info(&args.index_name);

    let mut files: AHashSet<_> = std::fs::read_dir(&args.folder)
        .unwrap()
        .map(|file| file.unwrap().path())
        .collect();

    if args.recursive {
        files = files
            .into_iter()
            .map(|dir| std::fs::read_dir(dir).unwrap())
            .flatten()
            .map(|f| f.unwrap().path())
            .collect();
    }

    let files: Vec<_> = files.difference(&info.indexed_files).cloned().collect();
    println!("{info:#?}");
    println!();
    println!("Files to Index: {files:#?}");
    println!();

    let b = std::time::Instant::now();
    let next_doc_id = AtomicU32::new(info.next_doc_id);
    let next_shard_id = AtomicU32::new(info.next_shard_id);

    files
    .par_chunks(args.files_per_index)
    .for_each(|files| {
        let shard_id = next_shard_id.fetch_add(1, Relaxed);
        
        println!("Start: {shard_id}");
        let b = std::time::Instant::now();
        let db = DB::truncate(&args.index_name, shard_id, args.db_size);
        let indexer = db.indexer(&next_doc_id);
        let (docs_in_shard, last_doc_id) = indexer.index(files);
        let next_doc_id = next_doc_id.load(Relaxed);
        println!("End:   {shard_id} ({:?}) (#docs: {docs_in_shard} | last doc id: {last_doc_id} | next doc id: {next_doc_id})", b.elapsed());

        if next_doc_id >= args.max_doc_id {
            println!("Max doc id reached");
            return;
        }
    });
    println!();
    println!("Whole Indexing took {:?}", b.elapsed());
}

fn search(args: Search) {
    let b = std::time::Instant::now();

    let info = DB::get_shards_info(&args.index_name);

    let dbs: Vec<_> = info
    .shards_ok
    .par_iter()
    .map(|(shard_id, docs_in_shard)| {
        let db = DB::open(&args.index_name, *shard_id, args.db_size);
        println!("Loaded shard {shard_id}: {docs_in_shard} docs");
        db
    })
    .collect();

    println!();
    println!("Opening database took: {:?}", b.elapsed());
    println!();

    let queries = std::fs::read_to_string(args.queries).unwrap();

    let queries: Vec<_> = queries.lines().collect();

    for q in queries.iter() {
        let mut sum_micros = 0;
        let mut res = 0;
        for _ in 0..args.runs {
            Command::new("/bin/bash")
                .arg("./clear_cache.sh")
                .stdout(Stdio::null())
                .status()
                .unwrap();

            let b = std::time::Instant::now();

            let doc_ids: Vec<_> = dbs
            .par_iter()
            .map(|db| db.search(q))
            .flatten()
            .collect();
            sum_micros += b.elapsed().as_micros();
            res = res.max(doc_ids.len());
        }
        let avg_ms = sum_micros as f64 / args.runs as f64 / 1000 as f64;
        println!("query: {q:?} took {avg_ms:.4} ms/iter ({res} results found)");
    }
}

fn main() {
    let args = CommandArgs::parse();

    // spawn rayon threads to avoid unecessary respawn
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    match args.ty {
        Ty::IndexFolder(arg) => index_folder(arg),
        Ty::Search(arg) => search(arg),
    }
}
