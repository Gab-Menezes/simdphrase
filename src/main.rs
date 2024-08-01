#![feature(new_uninit)]

// use arrow::array::{Int32Array, StringArray};
// use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use clap::{Args, Parser, Subcommand, ValueEnum};
use phrase_search::{KeywordSearch, Roaringish, DB};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rkyv::{ser::serializers::AllocSerializer, Deserialize, Infallible, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicU32, Ordering::Relaxed},
};

use ahash::AHashSet;

#[derive(Parser, Debug)]
struct CommandArgs {
    #[command(subcommand)]
    ty: Ty,
}

#[derive(Subcommand, Debug)]
enum Ty {
    IndexText(IndexFolder),
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
    D: Serialize<AllocSerializer<256>> + 'static,
    D::Archived: Deserialize<D, Infallible>
{
    let b = std::time::Instant::now();

    let db = DB::<D>::open(&args.index_name, args.db_size);

    println!();
    println!("Opening database took: {:?}", b.elapsed());
    println!();

    let queries = std::fs::read_to_string(args.queries).unwrap();

    let queries: Vec<_> = queries.lines().collect();

    for q in queries.iter() {
        let mut sum_micros = 0;
        let mut res = 0;
        for _ in 0..args.runs {
            // Command::new("/bin/bash")
            //     .arg("./clear_cache.sh")
            //     .stdout(Stdio::null())
            //     .status()
            //     .unwrap();

            let b = std::time::Instant::now();

            let doc_ids: Vec<_> = db.search(q);
            // let doc_ids = db.get_documents_by_ids(doc_ids);

            // println!("{doc_ids:?}");
            sum_micros += b.elapsed().as_micros();
            res = res.max(doc_ids.len());
        }
        let avg_ms = sum_micros as f64 / args.runs as f64 / 1000 as f64;
        println!("query: {q:?} took {avg_ms:.4} ms/iter ({res} results found)");
    }
}

fn index_text(args: IndexFolder) {
    println!("Start Indexing");

    let files: Vec<_> = std::fs::read_dir(&args.folder)
        .unwrap()
        .map(|file| file.unwrap().path())
        .collect();

    println!();
    println!("Files to Index: {files:#?}");
    println!();

    let docs: Vec<_> = files
        .into_iter()
        .map(|file| {
            let content = std::fs::read_to_string(&file).unwrap();
            (content, file.to_str().unwrap().to_owned())
        })
        .collect();

    let b = std::time::Instant::now();
    let next_doc_id = AtomicU32::new(0);

    let db = DB::truncate(&args.index_name, args.db_size);
    let indexer = db.indexer(&next_doc_id);
    indexer.index(docs);

    println!("Whole Indexing took {:?}", b.elapsed());
}

// fn index_parquet(args: IndexFolder) {
//     println!("Start Indexing");

//     let files: Vec<_> = std::fs::read_dir(&args.folder)
//         .unwrap()
//         .map(|file| file.unwrap().path())
//         .collect();

//     let mut docs = Vec::new();
//     for file in files.iter() {
//         let file = File::open(file).unwrap();
//         let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

//         let reader = builder.build().unwrap();

//         for record_batch in reader {
//             let record_batch = record_batch.unwrap();
//             let query_id = record_batch.column(3);
//             let query_id = query_id.as_any().downcast_ref::<Int32Array>().unwrap();
//             let queries = record_batch.column(2);
//             let queries = queries.as_any().downcast_ref::<StringArray>().unwrap();
//             for (query_id, query) in query_id.iter().zip(queries.iter()) {
//                 docs.push((query.unwrap().to_owned(), query_id.unwrap()));
//             }
//         }
//     }

//     println!();
//     println!("Files to Index: {files:#?} ({} documents)", docs.len());
//     println!();

//     let b = std::time::Instant::now();
//     let next_doc_id = AtomicU32::new(0);

//     let db = DB::truncate(&args.index_name, args.db_size);
//     let indexer = db.indexer(&next_doc_id);
//     indexer.index(docs);

//     println!("Whole Indexing took {:?}", b.elapsed());
// }

fn index_msmarco(args: IndexFile) {
    println!("Start Indexing");

    let reader = BufReader::new(File::open(args.file).unwrap());
    let it = reader
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.unwrap();
            let mut it = line.split("\t");
            it.next().unwrap();
            it.next().unwrap();
            (it.next().unwrap().to_owned(), i as u32)
        });

    let b = std::time::Instant::now();
    let next_doc_id = AtomicU32::new(0);

    let db = DB::truncate(&args.index_name, args.db_size);
    let indexer = db.indexer(&next_doc_id);
    let num_docs = indexer.index(it);

    println!("Whole Indexing took {:?} (total number of documents: {num_docs})", b.elapsed());
}

fn main() {
    let args = CommandArgs::parse();

    // spawn rayon threads to avoid unecessary respawn
    // rayon::ThreadPoolBuilder::new().build_global().unwrap();

    match args.ty {
        Ty::IndexText(arg) => index_text(arg),
        // Ty::IndexParquet(arg) => index_parquet(arg),
        Ty::IndexMsMarco(arg) => index_msmarco(arg),
        Ty::Search(arg) => match arg.data_set {
            DataSet::Text => search::<String>(arg),
            DataSet::Parquet => search::<i32>(arg),
            DataSet::MsMarco => search::<u32>(arg),
        },
    }
}
