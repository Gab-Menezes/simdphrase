use bumpalo::Bump;
use gxhash::{HashMap as GxHashMap, HashMapExt};
use heed::{
    types::Str, Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn,
    Unspecified,
};
use memmap2::{Mmap, MmapMut};
use rkyv::{
    api::high::HighSerializer,
    de::Pool,
    deserialize,
    rancor::Strategy,
    ser::{allocator::ArenaHandle, writer::IoWriter},
    util::AlignedVec,
    with::InlineAsBox,
    Archive, Archived, Deserialize, Serialize,
};
use std::{
    cmp::Reverse,
    collections::{hash_map::Entry, BinaryHeap, HashSet},
    fmt::Debug,
    fs::File,
    hash::Hash,
    io::BufWriter,
    num::NonZero,
    ops::Index,
    path::Path,
    sync::atomic::Ordering::Relaxed,
};

use crate::{
    codecs::{NativeU32, ZeroCopyCodec},
    error::{DatabaseError, DocumentError, SearchError},
    normalize,
    roaringish::{Aligned, ArchivedBorrowRoaringishPacked, RoaringishPackedKind, Unaligned},
    stats::Stats,
    tokenize, BorrowRoaringishPacked, Intersection, RoaringishPacked,
};

struct Tokens {
    tokens: String,
    positions: Vec<(usize, usize)>,
}

impl Tokens {
    fn new(q: &str) -> Self {
        let q = normalize(q);
        let mut start = 0;
        let mut tokens = String::with_capacity(q.len() + 1);
        let mut positions = Vec::with_capacity(q.len() + 1);

        for token in tokenize(&q) {
            tokens.push_str(token);
            tokens.push(' ');

            let b = start;
            let e = b + token.len();
            start = e + 1;
            positions.push((b, e));
        }
        tokens.pop();

        Self { tokens, positions }
    }

    fn as_ref(&self) -> RefTokens {
        RefTokens {
            tokens: &self.tokens,
            positions: &self.positions,
        }
    }
}

#[derive(Clone, Copy)]
struct RefTokens<'a> {
    tokens: &'a str,
    positions: &'a [(usize, usize)],
}

impl RefTokens<'_> {
    fn len(&self) -> usize {
        self.positions.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn reserve_len(&self) -> usize {
        let n = MAX_WINDOW_LEN.get();
        let l = self.len();
        n * (l.max(n) - n + 1) + ((n - 1) * n) / 2
    }

    fn first(&self) -> Option<&str> {
        self.positions
            .first()
            .map(|(b, e)| unsafe { self.tokens.get_unchecked(*b..*e) })
    }

    fn ref_token_iter(&self) -> impl Iterator<Item = Self> + '_ {
        (0..self.positions.len()).map(|i| Self {
            tokens: self.tokens,
            positions: &self.positions[i..i + 1],
        })
    }

    fn iter(&self) -> impl Iterator<Item = &str> {
        self.positions
            .iter()
            .map(|(b, e)| unsafe { self.tokens.get_unchecked(*b..*e) })
    }

    fn range(&self) -> (usize, usize) {
        let (b, _) = self.positions.first().unwrap_or(&(0, 0));
        let (_, e) = self.positions.last().unwrap_or(&(0, 0));
        (*b, *e)
    }

    fn tokens(&self) -> &str {
        let (b, e) = self.range();
        unsafe { self.tokens.get_unchecked(b..e) }
    }

    fn split_at(&self, i: usize) -> (Self, Self) {
        let (l, r) = self.positions.split_at(i);
        (
            Self {
                tokens: self.tokens,
                positions: l,
            },
            Self {
                tokens: self.tokens,
                positions: r,
            },
        )
    }
}

impl PartialEq for RefTokens<'_> {
    fn eq(&self, other: &Self) -> bool {
        let t0 = self.tokens();
        let t1 = other.tokens();
        t0 == t1
    }
}

impl Eq for RefTokens<'_> {}

impl Hash for RefTokens<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tokens().hash(state);
    }
}

impl Index<usize> for RefTokens<'_> {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        let (b, e) = self.positions[index];
        unsafe { self.tokens.get_unchecked(b..e) }
    }
}

impl Debug for RefTokens<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefTokens")
            .field("tokens", &self.tokens())
            .field("positions", &self.positions)
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
struct RefTokenLinkedList<'a, 'alloc> {
    tokens: RefTokens<'a>,
    next: Option<&'alloc RefTokenLinkedList<'a, 'alloc>>,
}

impl<'a, 'alloc> RefTokenLinkedList<'a, 'alloc> {
    fn iter<'b: 'alloc>(&'b self) -> RefTokenLinkedListIter<'a, 'alloc> {
        RefTokenLinkedListIter(Some(self))
    }
}

struct RefTokenLinkedListIter<'a, 'alloc>(Option<&'alloc RefTokenLinkedList<'a, 'alloc>>);
impl<'a, 'alloc> Iterator for RefTokenLinkedListIter<'a, 'alloc> {
    type Item = &'alloc RefTokens<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0 {
            Some(linked_list) => {
                self.0 = linked_list.next;
                Some(&linked_list.tokens)
            }
            None => None,
        }
    }
}

#[derive(Archive, Serialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BorrowStr<'a>(#[rkyv(with = InlineAsBox)] &'a str);

mod db_constants {
    pub const DB_DOC_ID_TO_DOCUMENT: &str = "doc_id_to_document";
    pub const DB_TOKEN_TO_OFFSETS: &str = "token_to_offsets";
    pub const KEY_COMMON_TOKENS: &str = "common_tokens";
    pub const FILE_ROARINGISH_PACKED: &str = "roaringish_packed";
    pub const TEMP_FILE_TOKEN_TO_PACKED: &str = "temp_token_to_packed";
}

pub const MAX_WINDOW_LEN: NonZero<usize> = NonZero::new(3).unwrap();

#[derive(Debug, Serialize, Archive)]
struct Offset {
    begin: u64,
    len: u64,
}

/// Represents all types that can be stored in the database.
///
/// This basically means that the type must be serializable by [rkyv].
pub trait Document:
    for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
    + Archive
    + 'static
{
}
impl<D> Document for D where
    Self: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive
        + 'static
{
}

pub struct Db<D: Document> {
    pub env: Env,
    db_main: Database<Unspecified, Unspecified>,
    db_doc_id_to_document: Database<NativeU32, ZeroCopyCodec<D>>,
    db_token_to_offsets: Database<Str, ZeroCopyCodec<Offset>>,
}

unsafe impl<D: Document> Send for Db<D> {}

unsafe impl<D: Document> Sync for Db<D> {}

impl<D: Document> Db<D> {
    pub fn truncate<P: AsRef<Path>>(path: P, db_size: usize) -> Result<Self, DatabaseError> {
        let path = path.as_ref();
        let _ = std::fs::remove_dir_all(path);
        std::fs::create_dir_all(path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(2)
                .map_size(db_size)
                .flags(EnvFlags::WRITE_MAP | EnvFlags::MAP_ASYNC)
                .open(path)?
        };

        let mut wrtxn = env.write_txn()?;

        let db_main = env.create_database(&mut wrtxn, None)?;

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name(db_constants::DB_DOC_ID_TO_DOCUMENT)
            .create(&mut wrtxn)?;

        let db_token_to_offsets =
            env.create_database(&mut wrtxn, Some(db_constants::DB_TOKEN_TO_OFFSETS))?;

        wrtxn.commit()?;

        Ok(Self {
            env,
            db_main,
            db_doc_id_to_document,
            db_token_to_offsets,
        })
    }

    pub fn write_doc_id_to_document(
        &self,
        rwtxn: &mut RwTxn,
        doc_ids: &[u32],
        documents: &[D],
    ) -> Result<(), DatabaseError> {
        log::debug!("Writing documents");
        let b = std::time::Instant::now();
        for (doc_id, document) in doc_ids.iter().zip(documents.iter()) {
            self.db_doc_id_to_document
                .put_with_flags(rwtxn, PutFlags::APPEND, doc_id, document)?;
        }
        log::debug!("Writing documents took {:?}", b.elapsed());
        Ok(())
    }

    pub fn write_token_to_roaringish_packed(
        &self,
        token_to_token_id: &GxHashMap<Box<str>, u32>,
        token_id_to_roaringish_packed: &[RoaringishPacked],
        mmap_size: &mut usize,
        batch_id: u32,
    ) -> Result<(), DatabaseError> {
        log::debug!("Writing token to roaringish packed");
        let b = std::time::Instant::now();
        let mut token_to_packed: Vec<_> = token_to_token_id
            .iter()
            .map(|(token, token_id)| {
                let packed = &token_id_to_roaringish_packed[*token_id as usize];
                *mmap_size += packed.size_bytes();
                (BorrowStr(token), BorrowRoaringishPacked::new(packed))
            })
            .collect();
        token_to_packed.sort_unstable_by(|(token0, _), (token1, _)| token0.cmp(token1));

        let file_name = format!("{}_{batch_id}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
        let file = IoWriter::new(BufWriter::new(
            File::options()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(self.env.path().join(file_name))?,
        ));
        rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(&token_to_packed, file)?;
        log::debug!("Writing token to roaringish packed took {:?}", b.elapsed());
        Ok(())
    }

    pub fn generate_mmap_file(
        &self,
        number_of_distinct_tokens: u64,
        mmap_size: usize,
        number_of_batches: u32,
        rwtxn: &mut RwTxn,
    ) -> Result<(), DatabaseError> {
        #[inline(always)]
        unsafe fn write_to_mmap<const N: usize>(
            mmap: &mut MmapMut,
            mmap_offset: &mut usize,
            bytes: &[u8],
        ) -> Offset {
            let ptr = mmap.as_ptr().add(*mmap_offset);
            let offset = ptr.align_offset(N);

            *mmap_offset += offset;
            mmap[*mmap_offset..*mmap_offset + bytes.len()].copy_from_slice(bytes);

            let begin = *mmap_offset;
            *mmap_offset += bytes.len();
            Offset {
                begin: begin as u64,
                len: bytes.len() as u64,
            }
        }

        log::info!("Merging roaringish packed files to generate the final memory map file");
        let b = std::time::Instant::now();
        let file = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(self.env.path().join(db_constants::FILE_ROARINGISH_PACKED))?;
        let final_size = mmap_size as u64 + (number_of_distinct_tokens * 64);
        log::debug!("Creating file with size: {} bytes", final_size);
        file.set_len(final_size)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        let mut mmap_offset = 0;

        // we need to do this in 3 steps because of the borrow checker
        let files_mmaps = (0..number_of_batches)
            .map(|i| -> Result<Mmap, DatabaseError> {
                let file_name = format!("{}_{i}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
                let file = File::options()
                    .read(true)
                    .open(self.env.path().join(file_name))?;
                unsafe { Ok(Mmap::map(&file)?) }
            })
            .collect::<Result<Vec<_>, DatabaseError>>()?;
        let files_data: Vec<_> = files_mmaps
            .iter()
            .map(|mmap| unsafe {
                rkyv::access_unchecked::<
                    Archived<Vec<(BorrowStr<'_>, BorrowRoaringishPacked<'_, Unaligned>)>>,
                >(mmap)
            })
            .collect();
        let mut iters: Vec<_> = files_data
            .iter()
            .map(|tokens_to_packeds| tokens_to_packeds.iter())
            .collect();

        struct ToMerge<'a> {
            token: &'a ArchivedBorrowStr<'a>,
            packed: &'a ArchivedBorrowRoaringishPacked<'a, Unaligned>,
            i: usize,
        }
        impl PartialEq for ToMerge<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.token.0 == other.token.0 && self.i == other.i
            }
        }
        impl Eq for ToMerge<'_> {}
        impl PartialOrd for ToMerge<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for ToMerge<'_> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match self.token.0.cmp(&other.token.0) {
                    std::cmp::Ordering::Equal => self.i.cmp(&other.i),
                    ord => ord,
                }
            }
        }

        let mut heap = BinaryHeap::new();
        for (i, it) in iters.iter_mut().enumerate() {
            if let Some(token_to_packed) = it.next() {
                heap.push(Reverse(ToMerge {
                    token: &token_to_packed.0,
                    packed: &token_to_packed.1,
                    i,
                }))
            }
        }

        while let Some(token_to_packed) = heap.pop() {
            let to_merge = token_to_packed.0;
            if let Some(token_to_packed) = iters[to_merge.i].next() {
                heap.push(Reverse(ToMerge {
                    token: &token_to_packed.0,
                    packed: &token_to_packed.1,
                    i: to_merge.i,
                }));
            }

            let mut packed_kind = RoaringishPackedKind::Archived(to_merge.packed);
            loop {
                let Some(next_to_merge) = heap.peek() else {
                    break;
                };

                if next_to_merge.0.token.0 != to_merge.token.0 {
                    break;
                }

                // This pop can't fail because we peeked before
                let next_to_merge = heap.pop().unwrap().0;
                if let Some(token_to_packed) = iters[next_to_merge.i].next() {
                    heap.push(Reverse(ToMerge {
                        token: &token_to_packed.0,
                        packed: &token_to_packed.1,
                        i: next_to_merge.i,
                    }));
                }

                let next_to_merge_kind = RoaringishPackedKind::Archived(next_to_merge.packed);
                packed_kind = packed_kind.concat(next_to_merge_kind);
            }

            if to_merge.token.0.len() > 511 {
                continue;
            }

            let packed = packed_kind.as_bytes();
            let offset = unsafe { write_to_mmap::<64>(&mut mmap, &mut mmap_offset, packed) };
            self.db_token_to_offsets.put_with_flags(
                rwtxn,
                PutFlags::APPEND,
                &to_merge.token.0,
                &offset,
            )?;
        }

        drop(iters);
        drop(files_data);
        drop(files_mmaps);

        log::debug!("Finished merging roaringish packed files");
        log::debug!("Removing old files");
        for i in 0..number_of_batches {
            let file_name = format!("{}_{i}", db_constants::TEMP_FILE_TOKEN_TO_PACKED);
            std::fs::remove_file(self.env.path().join(file_name))?;
        }
        log::info!("Whole merging process took {:?}", b.elapsed());

        Ok(())
    }

    fn read_common_tokens(
        rotxn: &RoTxn,
        db_main: Database<Unspecified, Unspecified>,
    ) -> Result<HashSet<Box<str>>, DatabaseError> {
        let k = db_main
            .remap_types::<Str, ZeroCopyCodec<HashSet<Box<str>>>>()
            .get(rotxn, db_constants::KEY_COMMON_TOKENS)?
            .ok_or_else(|| {
                DatabaseError::KeyNotFound(
                    db_constants::KEY_COMMON_TOKENS.to_string(),
                    "main".to_string(),
                )
            })?;

        Ok(deserialize::<_, rkyv::rancor::Error>(k)?)
    }

    pub fn write_common_tokens(
        &self,
        rwtxn: &mut RwTxn,
        common_tokens: &HashSet<Box<str>>,
    ) -> Result<(), DatabaseError> {
        log::debug!("Writing common tokens");
        let b = std::time::Instant::now();
        self.db_main
            .remap_types::<Str, ZeroCopyCodec<HashSet<Box<str>>>>()
            .put(rwtxn, db_constants::KEY_COMMON_TOKENS, common_tokens)?;
        log::debug!("Writing common tokens took {:?}", b.elapsed());
        Ok(())
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<(Self, HashSet<Box<str>>, Mmap), DatabaseError> {
        let path = path.as_ref();
        let env = unsafe {
            EnvOpenOptions::new()
                .max_dbs(2)
                .flags(EnvFlags::READ_ONLY)
                .open(path)?
        };

        let rotxn = env.read_txn()?;

        let db_main = env
            .open_database(&rotxn, None)?
            .ok_or_else(|| DatabaseError::DatabaseError("main".to_string()))?;

        let db_doc_id_to_document = env
            .database_options()
            .types::<NativeU32, ZeroCopyCodec<D>>()
            .flags(DatabaseFlags::REVERSE_KEY)
            .name(db_constants::DB_DOC_ID_TO_DOCUMENT)
            .open(&rotxn)?
            .ok_or_else(|| {
                DatabaseError::DatabaseError(db_constants::DB_DOC_ID_TO_DOCUMENT.to_string())
            })?;

        let db_token_to_offsets = env
            .open_database(&rotxn, Some(db_constants::DB_TOKEN_TO_OFFSETS))?
            .ok_or_else(|| {
                DatabaseError::DatabaseError(db_constants::DB_TOKEN_TO_OFFSETS.to_string())
            })?;

        let common_tokens = Self::read_common_tokens(&rotxn, db_main)?;

        rotxn.commit()?;

        let mmap_file = File::open(path.join(db_constants::FILE_ROARINGISH_PACKED))?;
        let mmap = unsafe { Mmap::map(&mmap_file)? };

        Ok((
            Self {
                env,
                db_main,
                db_doc_id_to_document,
                db_token_to_offsets,
            },
            common_tokens,
            mmap,
        ))
    }

    // This function neeeds to be inline never, for some reason inlining this
    // function makes some queries performance unpredictable
    #[inline(never)]
    fn merge_and_minimize_tokens<'a, 'b>(
        &self,
        rotxn: &RoTxn,
        tokens: RefTokens<'a>,
        common_tokens: &HashSet<Box<str>>,
        mmap: &'b Mmap,

        bump: &Bump,
    ) -> Result<
        (
            Vec<RefTokens<'a>>,
            GxHashMap<RefTokens<'a>, BorrowRoaringishPacked<'b, Aligned>>,
        ),
        SearchError,
    > {
        #[inline(always)]
        fn check_before_recursion<'a, 'b, 'alloc, D: Document>(
            me: &Db<D>,
            rotxn: &RoTxn,
            tokens: RefTokens<'a>,
            token_to_packed: &mut GxHashMap<RefTokens<'a>, BorrowRoaringishPacked<'b, Aligned>>,
            mmap: &'b Mmap,
            memo_token_to_score_choices: &mut GxHashMap<
                RefTokens<'a>,
                (usize, &'alloc RefTokenLinkedList<'a, 'alloc>),
            >,
            bump: &'alloc Bump,
        ) -> Result<Option<usize>, SearchError> {
            if tokens.len() != 1 {
                return Ok(None);
            }

            let score = match token_to_packed.entry(tokens) {
                Entry::Occupied(e) => e.get().len(),
                Entry::Vacant(e) => {
                    let packed = me.roaringish_packed(rotxn, &tokens[0], mmap)?;
                    let score = packed.len();
                    e.insert(packed);

                    let linked_list = bump.alloc(RefTokenLinkedList { tokens, next: None });
                    memo_token_to_score_choices.insert(tokens, (score, linked_list));
                    score
                }
            };
            Ok(Some(score))
        }

        #[allow(clippy::too_many_arguments)]
        fn inner_merge_and_minimize_tokens<'a, 'b, 'alloc, D: Document>(
            me: &Db<D>,
            rotxn: &RoTxn,
            tokens: RefTokens<'a>,
            common_tokens: &HashSet<Box<str>>,
            token_to_packed: &mut GxHashMap<RefTokens<'a>, BorrowRoaringishPacked<'b, Aligned>>,
            mmap: &'b Mmap,
            memo_token_to_score_choices: &mut GxHashMap<
                RefTokens<'a>,
                (usize, &'alloc RefTokenLinkedList<'a, 'alloc>),
            >,

            bump: &'alloc Bump,
        ) -> Result<usize, SearchError> {
            const { assert!(MAX_WINDOW_LEN.get() == 3) };
            let mut final_score = usize::MAX;
            let mut best_token_choice = None;
            let mut best_rem_choice = None;

            // TODO: fix this, it looks ugly
            let mut end = tokens
                .iter()
                .skip(1)
                .take(MAX_WINDOW_LEN.get() - 1)
                .take_while(|t| common_tokens.contains(*t))
                .count()
                + 2;
            if common_tokens.contains(&tokens[0]) {
                end += 1;
            }
            end = end.min(MAX_WINDOW_LEN.get() + 1).min(tokens.len() + 1);

            for i in (1..end).rev() {
                let (tokens, rem) = tokens.split_at(i);

                let score = match token_to_packed.entry(tokens) {
                    Entry::Occupied(e) => e.get().len(),
                    Entry::Vacant(e) => {
                        let packed = me.roaringish_packed(rotxn, tokens.tokens(), mmap)?;
                        let score = packed.len();
                        e.insert(packed);
                        score
                    }
                };

                let mut rem_score = 0;
                if !rem.is_empty() {
                    rem_score = match memo_token_to_score_choices.get(&rem) {
                        Some(r) => r.0,
                        None => {
                            match check_before_recursion(
                                me,
                                rotxn,
                                rem,
                                token_to_packed,
                                mmap,
                                memo_token_to_score_choices,
                                bump,
                            )? {
                                Some(score) => score,
                                None => inner_merge_and_minimize_tokens(
                                    me,
                                    rotxn,
                                    rem,
                                    common_tokens,
                                    token_to_packed,
                                    mmap,
                                    memo_token_to_score_choices,
                                    bump,
                                )?,
                            }
                        }
                    };
                    if rem_score == 0 {
                        return Err(SearchError::MergeAndMinimizeNotPossible);
                    }
                }

                let calc_score = score + rem_score;
                if calc_score < final_score {
                    final_score = calc_score;

                    best_token_choice = Some(tokens);
                    if let Some((_, rem_choices)) = memo_token_to_score_choices.get(&rem) {
                        best_rem_choice = Some(*rem_choices);
                    };
                }
            }

            let choices = match (best_token_choice, best_rem_choice) {
                (None, None) => return Err(SearchError::MergeAndMinimizeNotPossible),
                (None, Some(_)) => return Err(SearchError::MergeAndMinimizeNotPossible),
                (Some(tokens), None) => bump.alloc(RefTokenLinkedList { tokens, next: None }),
                (Some(tokens), Some(rem)) => bump.alloc(RefTokenLinkedList {
                    tokens,
                    next: Some(rem),
                }),
            };

            memo_token_to_score_choices.insert(tokens, (final_score, choices));
            Ok(final_score)
        }

        // This function neeeds to be inline never, for some reason inlining this
        // function makes some queries performance unpredictable
        #[inline(never)]
        fn no_common_tokens<'a, 'b, D: Document>(
            me: &Db<D>,
            rotxn: &RoTxn,
            tokens: RefTokens<'a>,
            mmap: &'b Mmap,
        ) -> Result<
            (
                Vec<RefTokens<'a>>,
                GxHashMap<RefTokens<'a>, BorrowRoaringishPacked<'b, Aligned>>,
            ),
            SearchError,
        > {
            let l = tokens.len();
            let mut token_to_packed = GxHashMap::with_capacity(l);
            let mut v = Vec::with_capacity(l);

            for token in tokens.ref_token_iter() {
                let packed = me.roaringish_packed(rotxn, token.tokens(), mmap)?;
                token_to_packed.insert(token, packed);
                v.push(token);
            }

            Ok((v, token_to_packed))
        }

        if common_tokens.is_empty() {
            return no_common_tokens(self, rotxn, tokens, mmap);
        }

        let len = tokens.reserve_len();
        let mut memo_token_to_score_choices = GxHashMap::with_capacity(len);
        let mut token_to_packed = GxHashMap::with_capacity(len);

        let score = match check_before_recursion(
            self,
            rotxn,
            tokens,
            &mut token_to_packed,
            mmap,
            &mut memo_token_to_score_choices,
            bump,
        )? {
            Some(score) => score,
            None => inner_merge_and_minimize_tokens(
                self,
                rotxn,
                tokens,
                common_tokens,
                &mut token_to_packed,
                mmap,
                &mut memo_token_to_score_choices,
                bump,
            )?,
        };

        if score == 0 {
            return Err(SearchError::MergeAndMinimizeNotPossible);
        }
        match memo_token_to_score_choices.remove(&tokens) {
            Some((_, choices)) => {
                let v = choices.iter().copied().collect();
                Ok((v, token_to_packed))
            }
            None => Err(SearchError::MergeAndMinimizeNotPossible),
        }
    }

    fn roaringish_packed_from_offset<'a>(
        offset: &ArchivedOffset,
        mmap: &'a Mmap,
    ) -> Result<BorrowRoaringishPacked<'a, Aligned>, SearchError> {
        let begin = offset.begin.to_native() as usize;
        let len = offset.len.to_native() as usize;
        let end = begin + len;
        let Some(packed) = &mmap.get(begin..end) else {
            return Err(SearchError::InternalError);
        };
        let (l, packed, r) = unsafe { packed.align_to::<u64>() };
        if !l.is_empty() || !r.is_empty() {
            return Err(SearchError::InternalError);
        }

        mmap.advise_range(memmap2::Advice::Sequential, begin, len)
            .map_err(DatabaseError::from)?;

        Ok(BorrowRoaringishPacked::new_raw(packed))
    }

    #[inline(always)]
    pub fn roaringish_packed<'a>(
        &self,
        rotxn: &RoTxn,
        token: &str,
        mmap: &'a Mmap,
    ) -> Result<BorrowRoaringishPacked<'a, Aligned>, SearchError> {
        let offset = self
            .db_token_to_offsets
            .get(rotxn, token)
            .map_err(DatabaseError::from)?;
        match offset {
            Some(offset) => Self::roaringish_packed_from_offset(offset, mmap),
            None => Err(SearchError::TokenNotFound(token.to_string())),
        }
    }

    pub fn search<I: Intersection>(
        &self,
        q: &str,
        stats: &Stats,
        common_tokens: &HashSet<Box<str>>,
        mmap: &Mmap,
    ) -> Result<Vec<u32>, SearchError> {
        stats.iters.fetch_add(1, Relaxed);

        let b = std::time::Instant::now();
        let tokens = Tokens::new(q);
        let tokens = tokens.as_ref();
        stats
            .normalize_tokenize
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if tokens.is_empty() {
            return Err(SearchError::EmptyQuery);
        }

        let rotxn = self.env.read_txn().map_err(DatabaseError::from)?;
        if tokens.len() == 1 {
            // this can't failt, we just checked
            self.roaringish_packed(&rotxn, tokens.first().unwrap(), mmap)?
                .document_ids(stats);
        }

        let b = std::time::Instant::now();
        let bump = Bump::with_capacity(tokens.reserve_len() * 5);
        let (final_tokens, token_to_packed) =
            self.merge_and_minimize_tokens(&rotxn, tokens, common_tokens, mmap, &bump)?;
        stats
            .merge_minimize
            .fetch_add(b.elapsed().as_micros() as u64, Relaxed);

        if final_tokens.is_empty() {
            return Err(SearchError::EmptyQuery);
        }

        if final_tokens.len() == 1 {
            return token_to_packed
                .get(&final_tokens[0])
                .ok_or_else(|| SearchError::TokenNotFound(final_tokens[0].tokens().to_string()))
                .map(|p| p.document_ids(stats));
        }

        // at this point we know that we have at least
        // 2 tokens, so the loop will run at least once
        // changing the value of `i` to be inbounds
        let mut min = usize::MAX;
        let mut i = usize::MAX;
        for (j, ts) in final_tokens.array_windows::<2>().enumerate() {
            let l0 = token_to_packed
                .get(&ts[0])
                .ok_or_else(|| SearchError::TokenNotFound(ts[0].tokens().to_string()))?
                .len();

            let l1 = token_to_packed
                .get(&ts[1])
                .ok_or_else(|| SearchError::TokenNotFound(ts[1].tokens().to_string()))?
                .len();

            let l = l0 + l1;
            if l <= min {
                i = j;
                min = l;
            }
        }

        let lhs = &final_tokens[i];
        let mut lhs_len = lhs.len() as u32;
        let lhs = token_to_packed
            .get(lhs)
            .ok_or_else(|| SearchError::TokenNotFound(lhs.tokens().to_string()))?;

        let rhs = &final_tokens[i + 1];
        let mut rhs_len = rhs.len() as u32;
        let rhs = token_to_packed
            .get(rhs)
            .ok_or_else(|| SearchError::TokenNotFound(rhs.tokens().to_string()))?;

        let mut result = lhs.intersect::<I>(*rhs, lhs_len, stats);
        let mut result_borrow = BorrowRoaringishPacked::new(&result);

        let mut left_i = i.wrapping_sub(1);
        let mut right_i = i + 2;

        loop {
            let lhs = final_tokens.get(left_i);
            let rhs = final_tokens.get(right_i);
            match (lhs, rhs) {
                (Some(t_lhs), Some(t_rhs)) => {
                    let lhs = token_to_packed
                        .get(t_lhs)
                        .ok_or_else(|| SearchError::TokenNotFound(t_lhs.tokens().to_string()))?;
                    let rhs = token_to_packed
                        .get(t_rhs)
                        .ok_or_else(|| SearchError::TokenNotFound(t_rhs.tokens().to_string()))?;
                    if lhs.len() <= rhs.len() {
                        lhs_len += t_lhs.len() as u32;

                        result = lhs.intersect::<I>(result_borrow, lhs_len, stats);
                        result_borrow = BorrowRoaringishPacked::new(&result);

                        left_i = left_i.wrapping_sub(1);
                    } else {
                        result = result_borrow.intersect::<I>(*rhs, rhs_len, stats);
                        result_borrow = BorrowRoaringishPacked::new(&result);

                        lhs_len += rhs_len;
                        rhs_len = t_rhs.len() as u32;

                        right_i += 1;
                    }
                }
                (Some(t_lhs), None) => {
                    let lhs = token_to_packed
                        .get(t_lhs)
                        .ok_or_else(|| SearchError::TokenNotFound(t_lhs.tokens().to_string()))?;
                    lhs_len += t_lhs.len() as u32;

                    result = lhs.intersect::<I>(result_borrow, lhs_len, stats);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    left_i = left_i.wrapping_sub(1);
                }
                (None, Some(t_rhs)) => {
                    let rhs = token_to_packed
                        .get(t_rhs)
                        .ok_or_else(|| SearchError::TokenNotFound(t_rhs.tokens().to_string()))?;

                    result = result_borrow.intersect::<I>(*rhs, rhs_len, stats);
                    result_borrow = BorrowRoaringishPacked::new(&result);

                    lhs_len += rhs_len;
                    rhs_len = t_rhs.len() as u32;

                    right_i += 1;
                }
                (None, None) => break,
            }

            if result.is_empty() {
                return Err(SearchError::EmptyIntersection);
            }
        }

        Ok(result_borrow.document_ids(stats))
    }

    fn inner_get_archived_document<'a>(
        &self,
        rotxn: &'a RoTxn,
        doc_id: &u32,
    ) -> Result<&'a D::Archived, DocumentError> {
        self.db_doc_id_to_document
            .get(rotxn, doc_id)
            .map_err(DatabaseError::from)?
            .ok_or(DocumentError::DocumentNotFound(*doc_id))
    }

    pub fn archived_documents(
        &self,
        doc_ids: &[u32],
        cb: impl FnOnce(Vec<&D::Archived>),
    ) -> Result<(), DocumentError> {
        let rotxn = self.env.read_txn().map_err(DatabaseError::from)?;
        let docs = doc_ids
            .iter()
            .map(|doc_id| self.inner_get_archived_document(&rotxn, doc_id))
            .collect::<Result<Vec<_>, _>>()?;

        cb(docs);

        Ok(())
    }

    pub fn archived_document(
        &self,
        doc_id: u32,
        cb: impl FnOnce(&D::Archived),
    ) -> Result<(), DocumentError> {
        let rotxn = self.env.read_txn().map_err(DatabaseError::from)?;
        let doc = self.inner_get_archived_document(&rotxn, &doc_id)?;

        cb(doc);

        Ok(())
    }

    pub fn documents(&self, doc_ids: &[u32]) -> Result<Vec<D>, DocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        let rotxn = self.env.read_txn().map_err(DatabaseError::from)?;
        doc_ids
            .iter()
            .map(|doc_id| {
                let archived = self.inner_get_archived_document(&rotxn, doc_id)?;
                rkyv::deserialize::<D, rkyv::rancor::Error>(archived)
                    .map_err(|e| DocumentError::DatabaseError(DatabaseError::from(e)))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn document(&self, doc_id: u32) -> Result<D, DocumentError>
    where
        <D as Archive>::Archived: Deserialize<D, Strategy<Pool, rkyv::rancor::Error>>,
    {
        let rotxn = self.env.read_txn().map_err(DatabaseError::from)?;
        let archived = self.inner_get_archived_document(&rotxn, &doc_id)?;
        rkyv::deserialize::<D, rkyv::rancor::Error>(archived)
            .map_err(|e| DocumentError::DatabaseError(DatabaseError::from(e)))
    }
}
