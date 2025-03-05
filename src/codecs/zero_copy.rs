use std::{borrow::Cow, marker::PhantomData};

use rkyv::{
    Archive, Archived, Serialize, api::high::HighSerializer, ser::allocator::ArenaHandle,
    util::AlignedVec,
};

pub struct ZeroCopyCodec<T>(PhantomData<T>)
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>>
        + Archive;

impl<'a, T> heed::BytesEncode<'a> for ZeroCopyCodec<T>
where
    T: for<'b> Serialize<HighSerializer<AlignedVec, ArenaHandle<'b>, rkyv::rancor::Error>>
        + Archive
        + 'a,
{
    type EItem = T;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, heed::BoxedError> {
        let bytes = rkyv::to_bytes(item).map(|bytes| Cow::Owned(bytes.to_vec()));

        Ok(bytes?)
    }
}

impl<'a, T> heed::BytesDecode<'a> for ZeroCopyCodec<T>
where
    T: for<'b> Serialize<HighSerializer<AlignedVec, ArenaHandle<'b>, rkyv::rancor::Error>>
        + Archive
        + 'a,
{
    type DItem = &'a T::Archived;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, heed::BoxedError> {
        unsafe { Ok(rkyv::access_unchecked::<Archived<T>>(bytes)) }
    }
}
