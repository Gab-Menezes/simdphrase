use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use phrase_search::Roaringish;

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
    let mut group = c.benchmark_group("roaringish");

    group
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5));

    let pos = vec![
        (
            "small",
            vec![
                ("8/1", gen(8, 1)),
                ("4/2", gen(4, 2)),
                ("3/7", gen(3, 7)),
                ("7/3", gen(7, 3)),
            ],
        ),
        (
            "medium",
            vec![
                ("16/2", gen(16, 2)),
                ("32/2", gen(32, 2)),
                ("17/3", gen(17, 3)),
                ("7/12", gen(7, 12)),
            ],
        ),
        (
            "big",
            vec![
                ("300/4", gen(300, 4)),
                ("1000/16", gen(100, 16)),
                ("3677/7", gen(3677, 7)),
                ("16000/16", gen(16000, 16)),
            ],
        ),
    ];

    for (id, pos) in pos {
        for (l, pos) in pos {
            group.bench_function(format!("{id}/{l}"), |b| {
                b.iter_batched(
                    || pos.clone(),
                    |pos| Roaringish::from_positions_sorted(pos),
                    BatchSize::SmallInput,
                )
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
