#![feature(portable_simd)]
#![feature(maybe_uninit_slice)]
#![feature(avx512_target_feature)]
#![feature(stdarch_x86_avx512)]
#![feature(pointer_is_aligned_to)]
#![feature(allocator_api)]
#![feature(iter_intersperse)]
#![feature(new_zeroed_alloc)]

// use arrow::array::{Int32Array, StringArray};
// use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use clap::{Args, Parser, Subcommand, ValueEnum};
use heed::{types::Str, Database, DatabaseFlags, EnvFlags, EnvOpenOptions};
use core::str;
use phrase_search::{SimdIntersect, CommonTokens, Indexer, Searcher, Stats};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

#[derive(Parser, Debug)]
struct CommandArgs {
    #[command(subcommand)]
    ty: Ty,
}

#[derive(Subcommand, Debug)]
enum Ty {
    // IndexText(IndexFolder),
    // IndexParquet(IndexFolder),
    IndexMsMarco(IndexFile),
    Search(Search),
}

#[derive(Args, Debug)]
struct IndexFolder {
    folder: PathBuf,
    index_name: PathBuf,
    db_size: usize,

    #[arg(short, long)]
    recursive: bool,
}

#[derive(Args, Debug)]
struct IndexFile {
    file: PathBuf,
    index_name: PathBuf,
    db_size: usize,

    #[arg(short, long)]
    recursive: bool,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum DataSet {
    Text,
    Parquet,
    MsMarco,
}

#[derive(Args, Debug)]
struct Search {
    runs: u32,
    queries: PathBuf,
    index_name: PathBuf,
    db_size: usize,
    data_set: DataSet,
}

fn search<D>(args: Search)
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Send
        + Sync
        + 'static,
{
    type Intersect = SimdIntersect;

    let searcher = Searcher::<D>::new(&args.index_name).unwrap();

    let queries = std::fs::read_to_string(args.queries).unwrap();
    let queries: Vec<_> = queries.lines().collect();
    for q in queries.iter() {
        for _ in 0..20 {
            let doc_ids = searcher.search::<Intersect>(q);
            std::hint::black_box(doc_ids);
        }

        let stats = Stats::default();
        let b = std::time::Instant::now();
        for _ in 0..args.runs {

            let doc_ids = searcher.search_with_stats::<Intersect>(q, &stats);
            std::hint::black_box(doc_ids);
        }
        let e = b.elapsed().as_micros();
        let avg_ms = e as f64 / args.runs as f64 / 1000_f64;
        let n_found = searcher.search::<Intersect>(q).len();

        println!("query: {q:?} took {avg_ms:.4} ms/iter ({n_found} results found)");
        println!("{stats:#?}");
    }
}

fn index_msmarco(args: IndexFile) {
    println!("Start Indexing");

    let reader = BufReader::new(File::open(args.file).unwrap());
    let it = reader.lines().enumerate().map(|(i, line)| {
        let line = line.unwrap();
        let mut it = line.split("\t");
        it.next().unwrap();
        it.next().unwrap();
        (it.next().unwrap().to_owned(), i as u32)
    });

    let b = std::time::Instant::now();
    let indexer = Indexer::new(Some(30000), Some(CommonTokens::FixedNum(50)));
    let num_docs = indexer.index(it, &args.index_name, args.db_size).unwrap();
    println!(
        "Whole Indexing took {:?} ({num_docs} documents)",
        b.elapsed()
    );
}

fn main() {
    //0.0002f64
    let args = CommandArgs::parse();

    // spawn rayon threads to avoid unecessary respawn
    // rayon::ThreadPoolBuilder::new().build_global().unwrap();

    match args.ty {
        // Ty::IndexText(arg) => index_text(arg),
        // Ty::IndexParquet(arg) => index_parquet(arg),
        Ty::IndexMsMarco(arg) => index_msmarco(arg),
        Ty::Search(arg) => match arg.data_set {
            DataSet::Text => search::<String>(arg),
            DataSet::Parquet => search::<i32>(arg),
            DataSet::MsMarco => search::<u32>(arg),
        },
    }
}
