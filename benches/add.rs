use std::num::NonZero;

use arbitrary::{Arbitrary, Unstructured};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use phrase_search::{AlignedBorrowRoaringishPacked, AlignedRoaringishPacked};
use rand::{distributions::Standard, Rng, SeedableRng};

fn gen(num_groups: u32, per_group: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(8);
    for i in 0..num_groups {
        let group = i * 16;
        for i in 0..per_group {
            v.push(group + i);
        }
    }
    v
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(69420);

    let mut group = c.benchmark_group("add");

    group
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5));

    let packed: Vec<_> = (0..10)
        .into_iter()
        .map(|_| {
            'a: loop {
                let data: Vec<NonZero<u8>> = (0..262144).into_iter().map(|_| rng.gen()).collect();
                let data = unsafe { std::mem::transmute::<Vec<NonZero<u8>>, Vec<u8>>(data) };
                let mut dna = Unstructured::new(&data);
                if let Ok(p) = AlignedRoaringishPacked::arbitrary(&mut dna) {
                    if p.len() < 10 {
                        continue;
                    }
                    println!("{}", p.len());
                    break 'a p;
                }
            }
        })
        .collect();

    let packed: Vec<_> = packed
        .iter()
        .map(|p| {
            AlignedBorrowRoaringishPacked::new(p)
        })
        .collect();

    let lens = [1u64, 2, 3, 4, 8, 12, 15];

    for (i, packed) in packed.into_iter().enumerate() {
        for len in lens {
            group.bench_function(format!("{i}/{len}/u64"), |b| {
                b.iter(
                    || std::hint::black_box(&packed + len),
                )
            });

            group.bench_function(format!("{i}/{len}/u32"), |b| {
                b.iter(
                    || std::hint::black_box(&packed + len as u32),
                )
            });
        }
    }

    // for (id, pos) in pos {
    //     for (l, pos) in pos {
    //         group.bench_function(format!("{id}/{l}"), |b| {
    //             b.iter_batched(
    //                 || pos.clone(),
    //                 |pos| Roaringish::from_positions_sorted(pos),
    //                 BatchSize::SmallInput,
    //             )
    //         });
    //     }
    // }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
