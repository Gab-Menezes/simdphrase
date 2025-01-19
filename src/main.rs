#![feature(portable_simd)]
#![feature(maybe_uninit_slice)]
#![feature(avx512_target_feature)]
#![feature(stdarch_x86_avx512)]
#![feature(pointer_is_aligned_to)]
#![feature(allocator_api)]
#![feature(iter_intersperse)]
#![feature(new_zeroed_alloc)]
#![feature(file_buffered)]

// use arrow::array::{Int32Array, StringArray};
// use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use clap::{Args, Parser, Subcommand, ValueEnum};
// use serde::Serialize as SerdeSerialize;
use core::str;
use phrase_search::{naive::NaiveIntersect, simd::SimdIntersect, tokenize, CommonTokens, Indexer, Intersect, Searcher, Stats};
use rkyv::{
    api::high::HighSerializer, ser::allocator::ArenaHandle, util::AlignedVec, Archive, Serialize,
};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
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

    let searcher = Searcher::<D>::new(&args.index_name, args.db_size).unwrap();

    let queries = std::fs::read_to_string(args.queries).unwrap();
    let queries: Vec<_> = queries.lines().collect();
    for q in queries.iter() {
        let stats = Stats::default();
        for _ in 0..20 {
            let doc_ids = searcher.search::<Intersect>(q, &stats);
            std::hint::black_box(doc_ids);
        }

        let stats = Stats::default();
        let b = std::time::Instant::now();
        for _ in 0..args.runs {
            // Command::new("/bin/bash")
            //     .arg("./clear_cache.sh")
            //     .stdout(Stdio::null())
            //     .status()
            //     .unwrap();

            let doc_ids = searcher.search::<Intersect>(q, &stats);
            std::hint::black_box(doc_ids);
        }
        let e = b.elapsed().as_micros();
        let avg_ms = e as f64 / args.runs as f64 / 1000_f64;

        let stats_ = Stats::default();
        let n_found = searcher.search::<Intersect>(q, &stats_).len();

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
    let num_docs = indexer.index(it, &args.index_name, args.db_size);
    println!(
        "Whole Indexing took {:?} ({num_docs} documents)",
        b.elapsed()
    );
}

fn benchmark<D>(args: Search)
where
    D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Send
        + Sync
        + 'static,
{
    fn inner<D, I>(runs: u32, q: &str, searcher: &Searcher<D>) -> (f64, usize)
    where
        D: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + Send
        + Sync
        + 'static,
        I: Intersect
    {
        let stats = Stats::default();
        for _ in 0..20 {
            let doc_ids = searcher.search::<I>(q, &stats);
            std::hint::black_box(doc_ids);
        }

        let stats = Stats::default();
        let b = std::time::Instant::now();
        for _ in 0..runs {
            let doc_ids = searcher.search::<I>(q, &stats);
            std::hint::black_box(doc_ids);
        }
        let e = b.elapsed().as_micros();
        let avg_ms = e as f64 / runs as f64 / 1000_f64;

        let stats_ = Stats::default();
        let n_found = searcher.search::<I>(q, &stats_).len();

        (avg_ms, n_found)
    }

    let searcher = Searcher::<D>::new(&args.index_name, args.db_size).unwrap();

    let queries = std::fs::read_to_string(args.queries).unwrap();
    let queries: Vec<_> = queries.lines().collect();
    for q in queries.iter() {
        let (t0, r0) = inner::<_, SimdIntersect>(args.runs, q, &searcher);
        let (t1, r1) = inner::<_, NaiveIntersect>(args.runs, q, &searcher);
        assert_eq!(r0, r1);
        println!("{q} | {t0:.4} | {t1:.4} | {r0}");
    }
}

fn main() {
    // #[derive(SerdeSerialize)]
    // struct Doc<'a> {
    //     id: u32,
    //     content: &'a str
    // }

    // let file = File::create("./docs.jsonl").unwrap();
    // let mut writer = BufWriter::with_capacity(32 * 1024, file);
    // let reader = BufReader::new(File::open("./fulldocs.tsv").unwrap());
    // for (id, line) in reader.lines().enumerate() {
    //     let line = line.unwrap();
    //     let mut it = line.split("\t");
    //     it.next().unwrap();
    //     it.next().unwrap();

    //     let content = it.next().unwrap();
    //     let doc = Doc {
    //         content,
    //         id: id as u32
    //     };
    //     serde_json::to_writer(&mut writer, &doc).unwrap();
    //     writer.write_all(&[b'\n']).unwrap();
    //     if id % 50_000 == 0{
    //         println!("{id}");
    //     }
    // }

    // writer.flush().unwrap();

    // return;
    //0.0002f64
    let args = CommandArgs::parse();

    println!("Query             | Simd    | Naive  | Results");
    println!("------------------| --------| -------| -------");

    // spawn rayon threads to avoid unecessary respawn
    // rayon::ThreadPoolBuilder::new().build_global().unwrap();

    match args.ty {
        // Ty::IndexText(arg) => index_text(arg),
        // Ty::IndexParquet(arg) => index_parquet(arg),
        Ty::IndexMsMarco(arg) => index_msmarco(arg),
        Ty::Search(arg) => match arg.data_set {
            DataSet::Text => search::<String>(arg),
            DataSet::Parquet => search::<i32>(arg),
            // DataSet::MsMarco => search::<u32>(arg),
            DataSet::MsMarco => benchmark::<u32>(arg),
        },
    }
}
