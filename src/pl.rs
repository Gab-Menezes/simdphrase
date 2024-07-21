use std::borrow::Cow;

use heed::{byteorder::BigEndian, types::U64, BoxedError};
use roaring::RoaringBitmap;

/// Some ideias for the PL
///
/// * Maybe make the positions to save memory
///     enum Positions {
///         Bitmap(Vec<RoaringBitmap>)
///         Vec(Vec<Vec<u32>>)
///     }
/// and we decoding the Vec variant we could
/// construct a Box<[&[u32]]> to avoid copying
#[derive(Clone, Default, Debug)]
pub struct PostingList {
    pub(crate) doc_ids: RoaringBitmap,
    pub(crate) positions: Vec<RoaringBitmap>,
}

impl PostingList {
    pub fn new(doc_ids: RoaringBitmap, positions: Vec<RoaringBitmap>) -> Self {
        Self {
            doc_ids,
            positions
        }
    }
}

pub struct PostingListCodec;

impl heed::BytesDecode<'_> for PostingListCodec {
    type DItem = PostingList;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, BoxedError> {
        const S: usize = std::mem::size_of::<u64>();
        let doc_ids_serialized_len = U64::<BigEndian>::bytes_decode(&bytes[0..S])? as usize;
        let doc_ids =
            RoaringBitmap::deserialize_unchecked_from(&bytes[S..(doc_ids_serialized_len + S)])?;

        let positions_len = U64::<BigEndian>::bytes_decode(
            &bytes[(doc_ids_serialized_len + S)..(doc_ids_serialized_len + S + S)],
        )? as usize;

        let mut begin = doc_ids_serialized_len + S + S;
        let mut positions = Vec::with_capacity(positions_len);
        for _ in 0..positions_len {
            let serialized_len =
                U64::<BigEndian>::bytes_decode(&bytes[begin..(begin + S)])? as usize;
            let position = RoaringBitmap::deserialize_unchecked_from(
                &bytes[(begin + S)..(begin + S + serialized_len)],
            )?;
            positions.push(position);
            begin += S + serialized_len;
        }

        Ok(PostingList { doc_ids, positions })
    }
}

impl heed::BytesEncode<'_> for PostingListCodec {
    type EItem = PostingList;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'_, [u8]>, BoxedError> {
        const S: usize = std::mem::size_of::<u64>();

        let serialized_len_sum: usize = item.positions.iter().map(|m| m.serialized_size()).sum();
        let doc_ids_size = item.doc_ids.serialized_size();

        let mut bytes =
            vec![0u8; S + doc_ids_size + S + S * item.doc_ids.len() as usize + serialized_len_sum];

        let doc_ids_serialized_len = item.doc_ids.serialized_size() as u64;
        let doc_ids_serialized_len_bytes = U64::<BigEndian>::bytes_encode(&doc_ids_serialized_len)?;

        let positions_len = item.positions.len() as u64;
        let positions_len_bytes = U64::<BigEndian>::bytes_encode(&positions_len)?;

        bytes[0..S].copy_from_slice(&doc_ids_serialized_len_bytes);
        item.doc_ids
            .serialize_into(&mut bytes[S..(S + doc_ids_serialized_len as usize)])?;
        bytes[(S + doc_ids_serialized_len as usize)..(S + doc_ids_serialized_len as usize + S)]
            .copy_from_slice(&positions_len_bytes);

        let mut begin = S + doc_ids_serialized_len as usize + S;
        for r in item.positions.iter() {
            let pos_serialized_len = r.serialized_size() as u64;
            let pos_serialized_len_bytes = U64::<BigEndian>::bytes_encode(&pos_serialized_len)?;
            bytes[begin..(begin + S)].copy_from_slice(&pos_serialized_len_bytes);
            r.serialize_into(&mut bytes[(begin + S)..(begin + S + pos_serialized_len as usize)])?;
            begin += S + pos_serialized_len as usize;
        }

        Ok(Cow::Owned(bytes))
    }
}