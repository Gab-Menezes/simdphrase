#![feature(new_uninit)]
#![feature(portable_simd)]
#![feature(maybe_uninit_slice)]

use ahash::AHashSet;
// use arrow::array::{Int32Array, StringArray};
// use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use clap::{Args, Parser, Subcommand, ValueEnum};
use phrase_search::{normalize, tokenize, CommonTokens, Indexer, Searcher, Stats};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rkyv::{
    rend::unaligned::u16_ule, ser::DefaultSerializer, util::AlignedVec, Archive, Deserialize,
    Serialize,
};
use std::{
    fmt::{Debug, Display},
    fs::File,
    io::{BufRead, BufReader},
    iter::Map,
    mem::MaybeUninit,
    path::PathBuf,
    process::{Command, Stdio},
    simd::{cmp::SimdPartialEq, Mask, Simd},
    sync::atomic::{AtomicU32, Ordering::Relaxed},
    arch::asm
};

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

fn search<D: Send + Sync>(args: Search)
where
    D: for<'a> Serialize<DefaultSerializer<'a, AlignedVec, rkyv::rancor::Error>>
        + Archive
        + 'static,
{
    let b = std::time::Instant::now();

    let searcher = Searcher::<D>::new(&args.index_name, args.db_size).unwrap();
    // let shard = searcher.get_shard(0).unwrap();

    println!();
    println!("Opening database took: {:?}", b.elapsed());
    println!();

    let queries = std::fs::read_to_string(args.queries).unwrap();

    let queries: Vec<_> = queries.lines().collect();

    for q in queries.iter() {
        let mut sum_micros = 0;
        let mut res = 0;
        let stats = Stats::default();
        for _ in 0..args.runs {
            // Command::new("/bin/bash")
            //     .arg("./clear_cache.sh")
            //     .stdout(Stdio::null())
            //     .status()
            //     .unwrap();

            let b = std::time::Instant::now();

            // searcher.foo(q);
            let doc_ids = searcher.search(q, &stats);
            // let doc_ids = shard.search(q, &stats);
            // println!("{doc_ids:?}");
            // let doc_ids = db.get_documents_by_ids(doc_ids);

            sum_micros += b.elapsed().as_micros();
            res = res.max(doc_ids.len());
        }
        let avg_ms = sum_micros as f64 / args.runs as f64 / 1000 as f64;
        println!("query: {q:?} took {avg_ms:.4} ms/iter ({res} results found)");
        println!("{stats:#?}");
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

    // let mut indexer = Indexer::new(&args.index_name, args.db_size, None, None, None);
    let mut indexer = Indexer::new(&args.index_name, args.db_size, None, None, None);
    let num_docs = indexer.index(docs);
    indexer.flush();

    println!(
        "Whole Indexing took {:?} ({num_docs} documents)",
        b.elapsed()
    );
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
    let it = reader.lines().enumerate().map(|(i, line)| {
        let line = line.unwrap();
        let mut it = line.split("\t");
        it.next().unwrap();
        it.next().unwrap();
        (it.next().unwrap().to_owned(), i as u32)
    });

    let b = std::time::Instant::now();

    let mut indexer = Indexer::new(
        &args.index_name,
        args.db_size,
        Some(500000),
        Some(CommonTokens::FixedNum(50)),
        Some(1460),
    );
    let num_docs = indexer.index(it);
    indexer.flush();

    println!(
        "Whole Indexing took {:?} ({num_docs} documents)",
        b.elapsed()
    );
}

#[inline(never)]
unsafe fn intersect(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    #[inline(always)]
    unsafe fn word_wise_check<F: Fn(u64) -> u64>(
        a: &[u64],
        lhs_i: usize,
        b: &[u64],
        rhs_i: usize,
        op: F,
    ) -> Mask<i32, 16> {
        let a0 = op(*a.get_unchecked(lhs_i + 0)) as u32;
        let a1 = op(*a.get_unchecked(lhs_i + 1)) as u32;
        let a2 = op(*a.get_unchecked(lhs_i + 2)) as u32;
        let a3 = op(*a.get_unchecked(lhs_i + 3)) as u32;
        let amask = Simd::from_array([
            a0, a0, a0, a0, a1, a1, a1, a1, a2, a2, a2, a2, a3, a3, a3, a3,
        ]);

        let b0 = op(*b.get_unchecked(rhs_i + 0)) as u32;
        let b1 = op(*b.get_unchecked(rhs_i + 1)) as u32;
        let b2 = op(*b.get_unchecked(rhs_i + 2)) as u32;
        let b3 = op(*b.get_unchecked(rhs_i + 3)) as u32;
        let bmask = Simd::from_array([
            b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3,
        ]);

        amask.simd_eq(bmask)
    }

    #[inline(always)]
    fn cmovnz(value: &mut MaybeUninit<u64>, v: &u64, condition: u8) {
        unsafe {
            asm! {
                "test {0}, {0}",
                "cmovnz {1:r}, {2:r}",
                in(reg_byte) condition,
                inlateout(reg) *value,
                in(reg) *v,
                options(pure, nomem, nostack),
            };
        }
    }

    macro_rules! advance {
        ($i:ident, $v:ident) => {
            $i += 4;
            if $i >= $v.len() {
                break;
            } else {
                continue;
            }
        };
    }

    macro_rules! advanceAB {
        ($lhs_i:ident, $a:ident, $rhs_i:ident, $b:ident) => {
            $lhs_i += 4;
            $rhs_i += 4;
            if $lhs_i >= $a.len() || $rhs_i >= $b.len() {
                break;
            } else {
                continue;
            }
        };
    }

    let min = lhs.len().min(rhs.len());
    let mut results: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
    let mut i = 0;

    let mut lhs_i = 0;
    let mut rhs_i = 0;

    let end_lhs = lhs.len() / 4 * 4;
    let end_rhs = rhs.len() / 4 * 4;
    let a = lhs.get_unchecked(..end_lhs);
    let b = rhs.get_unchecked(..end_rhs);
    loop {
        let lsb = word_wise_check(a, lhs_i, b, rhs_i, |v| v).to_bitmask();
        if lsb >= 1 {
            // this if is very likely to happend

            let msb = word_wise_check(a, lhs_i, b, rhs_i, |v| (v & 0xFFFFFFFF00000000) >> 32)
                .to_bitmask();
            // a0, a0, a0, a0 | a1, a1, a1, a1 | a2, a2, a2, a2 | a3, a3, a3, a3
            // b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3 | b0, b1, b2, b3
            let bitmask = lsb & msb;
            let a0b0123 = ((bitmask & 0b0000_0000_0000_1111) > 0) as u8;
            let a1b0123 = ((bitmask & 0b0000_0000_1111_0000) > 0) as u8;
            let a2b0123 = ((bitmask & 0b0000_1111_0000_0000) > 0) as u8;
            let b3 = (bitmask & 0b0000_1000_1000_1000) > 0;

            // let a3b0123 = ((bitmask & 0b1111_0000_0000_0000) > 0) as u8;
            let a3b012 = (bitmask & 0b0111_0000_0000_0000) > 0;
            let a3b3 = (bitmask & 0b1000_0000_0000_0000) > 0;


            let a0 = a.get_unchecked(lhs_i + 0);
            let a1 = a.get_unchecked(lhs_i + 1);
            let a2 = a.get_unchecked(lhs_i + 2);
            let a3 = a.get_unchecked(lhs_i + 3);

            cmovnz(results.get_unchecked_mut(i), a0, a0b0123);
            i += a0b0123 as usize;
            cmovnz(results.get_unchecked_mut(i), a1, a1b0123);
            i += a1b0123 as usize;
            cmovnz(results.get_unchecked_mut(i), a2, a2b0123);
            i += a2b0123 as usize;
            if b3 {
                advance!(rhs_i, b);
            }

            // cmovnz(results.get_unchecked_mut(i), a3, a3b0123);
            // i += a3b0123 as usize;
            if a3b012 {
                results.get_unchecked_mut(i).write(*a3);
                i += 1;
                advance!(lhs_i, a);
            } else if a3b3 {
                results.get_unchecked_mut(i).write(*a3);
                i += 1;
                advanceAB!(lhs_i, a, rhs_i, b);
            }

            // we hope that this if is not reached, to reduce
            // branch miss prediction
            if a3 > b.get_unchecked(rhs_i + 3) {
                advance!(rhs_i, b);
            } else {
                advance!(lhs_i, a);
            }
        } else if *a.get_unchecked(lhs_i + 3) > *b.get_unchecked(rhs_i + 3) {
            // we want to avoid this else if from running, since it's almost
            // a 50-50%, making it hard to predict, for this reason we repeat
            // this same check inside the 
            advance!(rhs_i, b);
        } else {
            advance!(lhs_i, a);
        }
    }

    // for the remaining elements we can do a 2 pointer approach
    // since the 2 slices will be small
    while lhs_i < lhs.len() && rhs_i < rhs.len() {
        if lhs[lhs_i] == rhs[rhs_i] {
            results.get_unchecked_mut(i).write(lhs[lhs_i]);
            i += 1;
            lhs_i += 1;
            rhs_i += 1;
        } else if lhs[lhs_i] > rhs[rhs_i] {
            rhs_i += 1;
        } else {
            lhs_i += 1;
        }
    }

    unsafe { Vec::from_raw_parts(Box::into_raw(results) as *mut _, i, min) }
}

fn naive(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let min = lhs.len().min(rhs.len());
    let mut results: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
    let mut i = 0;

    let mut lhs_i = 0;
    let mut rhs_i = 0;

    while lhs_i < lhs.len() && rhs_i < rhs.len() {
        if lhs[lhs_i] == rhs[rhs_i] {
            unsafe { results.get_unchecked_mut(i).write(lhs[lhs_i]) };
            i += 1;
            lhs_i += 1;
            rhs_i += 1;
        } else if lhs[lhs_i] > rhs[rhs_i] {
            rhs_i += 1;
        } else {
            lhs_i += 1;
        }
    }
    unsafe { Vec::from_raw_parts(Box::into_raw(results) as *mut _, i, min) }
}

fn gallop(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let min = lhs.len().min(rhs.len());
    let mut results: Box<[MaybeUninit<u64>]> = Box::new_uninit_slice(min);
    let mut i = 0;

    let mut lhs_i = 0;
    let mut rhs_i = 0;

    while lhs_i < lhs.len() && rhs_i < rhs.len() {
        let mut lhs_delta = 1;
        let mut rhs_delta = 1;

        while lhs_i < lhs.len() && lhs[lhs_i] < rhs[rhs_i] {
            lhs_i += lhs_delta;
            lhs_delta *= 2;
        }
        lhs_i -= lhs_delta / 2;

        while rhs_i < rhs.len() && rhs[rhs_i] < unsafe { *lhs.get_unchecked(lhs_i) } {
            rhs_i += rhs_delta;
            rhs_delta *= 2;
        }
        rhs_i -= rhs_delta / 2;

        let lhs = unsafe { lhs.get_unchecked(lhs_i) };
        let rhs = unsafe { rhs.get_unchecked(rhs_i) };
        match lhs.cmp(rhs) {
            std::cmp::Ordering::Less => lhs_i += 1,
            std::cmp::Ordering::Greater => rhs_i += 1,
            std::cmp::Ordering::Equal => {
                unsafe { results.get_unchecked_mut(i).write(*lhs) };
                i += 1;
                lhs_i += 1;
                rhs_i += 1;
            },
        }
    }
    unsafe { Vec::from_raw_parts(Box::into_raw(results) as *mut _, i, min) }
}

fn main() {
    let mut rng = rand::thread_rng();
    loop {
        let v0_len = rng.gen_range(500000usize..20000000);
        let mut v0: Vec<_> = (0..v0_len).map(|_| rng.gen_range(0u64..40000000)).collect();

        let v1_len = rng.gen_range(500000usize..20000000);
        let mut v1: Vec<_> = (0..v1_len).map(|_| rng.gen_range(0u64..40000000)).collect();

        v0.sort_unstable();
        v1.sort_unstable();
        v0.dedup();
        v1.dedup();

        const ITERS: usize = 128;
        let b = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(unsafe { intersect(&v0, &v1) });
        }
        let t_intersect = b.elapsed().as_millis() as f64 / ITERS as f64;

        let b = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(naive(&v0, &v1));
        }
        let t_naive = b.elapsed().as_millis() as f64 / ITERS as f64;

        let b = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(gallop(&v0, &v1));
        }
        let t_gallop = b.elapsed().as_millis() as f64 / ITERS as f64;

        println!("i/n: {:.4} | i/g: {:.4} | n/g: {:.4}", t_intersect / t_naive, t_intersect / t_gallop, t_naive / t_gallop);
        let int = unsafe { intersect(&v0, &v1) };
        let nai = naive(&v0, &v1);
        let gal = gallop(&v0, &v1);
        assert_eq!(int, nai, "{v0:?}\n\n{v1:?}");
        assert_eq!(int, gal, "{v0:?}\n\n{v1:?}");
    }
    return;
    // for e in 0..7 {
    //     for m in [1, 5] {
    //         let n = 10usize.pow(e) * m;
    //         let reader = BufReader::new(File::open("./fulldocs.tsv").unwrap());
    //         let mut len_sum = 0u64;
    //         let mut unique_tokens = AHashSet::new();
    //         let mut lines = 0u64;
    //         for line in reader.lines().take(n) {
    //             let line = line.unwrap();
    //             let mut it = line.split("\t");
    //             it.next().unwrap();
    //             it.next().unwrap();
    //             let content = it.next().unwrap();
    //             let content = normalize(content);
    //             lines += 1;
    //             for t in tokenize(&content) {
    //                 len_sum += 1;
    //                 unique_tokens.insert(t.to_owned());
    //             }
    //         }
    //         println!("{lines} {len_sum} {}", unique_tokens.len());
    //     }
    // }
    // return;
    // // for n in [1, 5, 10, 50, 100, 500, 1000, 5000]
    // // return;
    let args = CommandArgs::parse();

    // spawn rayon threads to avoid unecessary respawn
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

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
