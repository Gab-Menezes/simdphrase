use std::{
    fmt::Debug,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

/// Timing statistics collected during search.
#[derive(Default)]
pub struct Stats {
    /// Time spent during normalization and tokenization.
    pub normalize_tokenize: AtomicU64,
    /// Time spent during merging and minimizing.
    pub merge_minimize: AtomicU64,
    /// Time spent during the first binary search.
    pub first_binary_search: AtomicU64,
    /// Time spent during the first intersect.
    pub first_intersect: AtomicU64,
    /// Time spent during the first intersect using SIMD.
    pub first_intersect_simd: AtomicU64,
    /// Time spent during the first intersect using naive method.
    pub first_intersect_naive: AtomicU64,
    /// Time spent during the first intersect using gallop method.
    pub first_intersect_gallop: AtomicU64,

    /// Time spent during the second binary search.
    pub second_binary_search: AtomicU64,
    /// Time spent during the second intersect.
    pub second_intersect: AtomicU64,
    /// Time spent during the second intersect using SIMD.
    pub second_intersect_simd: AtomicU64,
    /// Time spent during the second intersect using naive method.
    pub second_intersect_naive: AtomicU64,
    /// Time spent during the second intersect using gallop method.
    pub second_intersect_gallop: AtomicU64,

    /// Time spent during the first merge phase.
    pub merge_phases_first_pass: AtomicU64,
    /// Time spent during the second merge phase.
    pub merge_phases_second_pass: AtomicU64,

    /// Time spent getting document ids.
    pub document_ids: AtomicU64,

    /// Number of calls to the search function.
    pub iters: AtomicU64,
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sum = self.normalize_tokenize.load(Relaxed)
            + self.merge_minimize.load(Relaxed)
            + self.first_binary_search.load(Relaxed)
            + self.first_intersect.load(Relaxed)
            + self.second_binary_search.load(Relaxed)
            + self.second_intersect.load(Relaxed)
            + self.merge_phases_first_pass.load(Relaxed)
            + self.merge_phases_second_pass.load(Relaxed)
            + self.document_ids.load(Relaxed);
        let sum = sum as f64;

        let normalize_tokenize = self.normalize_tokenize.load(Relaxed) as f64;
        let merge = self.merge_minimize.load(Relaxed) as f64;
        let first_binary_search = self.first_binary_search.load(Relaxed) as f64;
        let first_intersect = self.first_intersect.load(Relaxed) as f64;
        let first_intersect_simd = self.first_intersect_simd.load(Relaxed) as f64;
        let first_intersect_naive = self.first_intersect_naive.load(Relaxed) as f64;
        let first_intersect_gallop = self.first_intersect_gallop.load(Relaxed) as f64;
        let second_binary_search = self.second_binary_search.load(Relaxed) as f64;
        let second_intersect = self.second_intersect.load(Relaxed) as f64;
        let second_intersect_simd = self.second_intersect_simd.load(Relaxed) as f64;
        let second_intersect_naive = self.second_intersect_naive.load(Relaxed) as f64;
        let second_intersect_gallop = self.second_intersect_gallop.load(Relaxed) as f64;
        let merge_phases_first_pass = self.merge_phases_first_pass.load(Relaxed) as f64;
        let merge_phases_second_pass = self.merge_phases_second_pass.load(Relaxed) as f64;
        let document_ids = self.document_ids.load(Relaxed) as f64;
        let iters = self.iters.load(Relaxed) as f64;

        let per_normalize_tokenize = normalize_tokenize / sum * 100f64;
        let per_merge = merge / sum * 100f64;
        let per_first_binary_search = first_binary_search / sum * 100f64;
        let per_first_intersect = first_intersect / sum * 100f64;
        let per_second_binary_search = second_binary_search / sum * 100f64;
        let per_second_intersect = second_intersect / sum * 100f64;
        let per_merge_phases_first_pass = merge_phases_first_pass / sum * 100f64;
        let per_merge_phases_second_pass = merge_phases_second_pass / sum * 100f64;
        let per_document_ids = document_ids / sum * 100f64;

        f.debug_struct("Stats")
            .field(
                "normalize_tokenize",
                &format_args!(
                    "        ({:08.3}ms, {:08.3}us/iter, {per_normalize_tokenize:06.3}%)",
                    normalize_tokenize / 1000f64,
                    normalize_tokenize / iters,
                ),
            )
            .field(
                "merge_minimize",
                &format_args!(
                    "            ({:08.3}ms, {:08.3}us/iter, {per_merge:06.3}%)",
                    merge / 1000f64,
                    merge / iters,
                ),
            )
            .field(
                "first_binary_search",
                &format_args!(
                    "       ({:08.3}ms, {:08.3}us/iter, {per_first_binary_search:06.3}%)",
                    first_binary_search / 1000f64,
                    first_binary_search / iters,
                ),
            )
            .field(
                "first_intersect",
                &format_args!(
                    "           ({:08.3}ms, {:08.3}us/iter, {per_first_intersect:06.3}%)",
                    first_intersect / 1000f64,
                    first_intersect / iters,
                ),
            )
            .field(
                "    first_intersect_simd",
                &format_args!(
                    "      ({:08.3}ms, {:08.3}us/iter)",
                    first_intersect_simd / 1000f64,
                    first_intersect_simd / iters,
                ),
            )
            .field(
                "    first_intersect_naive",
                &format_args!(
                    "     ({:08.3}ms, {:08.3}us/iter)",
                    first_intersect_naive / 1000f64,
                    first_intersect_naive / iters,
                ),
            )
            .field(
                "    first_intersect_gallop",
                &format_args!(
                    "    ({:08.3}ms, {:08.3}us/iter)",
                    first_intersect_gallop / 1000f64,
                    first_intersect_gallop / iters,
                ),
            )
            .field(
                "second_binary_search",
                &format_args!(
                    "      ({:08.3}ms, {:08.3}us/iter, {per_second_binary_search:06.3}%)",
                    second_binary_search / 1000f64,
                    second_binary_search / iters,
                ),
            )
            .field(
                "second_intersect",
                &format_args!(
                    "          ({:08.3}ms, {:08.3}us/iter, {per_second_intersect:06.3}%)",
                    second_intersect / 1000f64,
                    second_intersect / iters,
                ),
            )
            .field(
                "    second_intersect_simd",
                &format_args!(
                    "     ({:08.3}ms, {:08.3}us/iter)",
                    second_intersect_simd / 1000f64,
                    second_intersect_simd / iters,
                ),
            )
            .field(
                "    second_intersect_naive",
                &format_args!(
                    "    ({:08.3}ms, {:08.3}us/iter)",
                    second_intersect_naive / 1000f64,
                    second_intersect_naive / iters,
                ),
            )
            .field(
                "    second_intersect_gallop",
                &format_args!(
                    "   ({:08.3}ms, {:08.3}us/iter)",
                    second_intersect_gallop / 1000f64,
                    second_intersect_gallop / iters,
                ),
            )
            .field(
                "merge_phases_first_pass",
                &format_args!(
                    "   ({:08.3}ms, {:08.3}us/iter, {per_merge_phases_first_pass:06.3}%)",
                    merge_phases_first_pass / 1000f64,
                    merge_phases_first_pass / iters,
                ),
            )
            .field(
                "merge_phases_second_pass",
                &format_args!(
                    "  ({:08.3}ms, {:08.3}us/iter, {per_merge_phases_second_pass:06.3}%)",
                    merge_phases_second_pass / 1000f64,
                    merge_phases_second_pass / iters,
                ),
            )
            .field(
                "document_ids",
                &format_args!(
                    "               ({:08.3}ms, {:08.3}us/iter, {per_document_ids:06.3}%)",
                    document_ids / 1000f64,
                    document_ids / iters,
                ),
            )
            .finish()
    }
}
