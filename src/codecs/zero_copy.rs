use std::{borrow::Cow, marker::PhantomData};

use rkyv::{rancor::{Error, Fallible}, ser::{DefaultSerializer, Serializer}, util::AlignedVec, Archive, Archived, Serialize};

pub struct ZeroCopyCodec<T>(PhantomData<T>)
where
    T: for<'a> Serialize<DefaultSerializer<'a, AlignedVec, rkyv::rancor::Error>> + Archive;

impl<'a, T> heed::BytesEncode<'a> for ZeroCopyCodec<T>
where
    T: for<'b> Serialize<DefaultSerializer<'b, AlignedVec, rkyv::rancor::Error>> + Archive + 'a
{
    type EItem = T;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, heed::BoxedError> {
        let bytes = rkyv::to_bytes(item).unwrap();
        Ok(Cow::Owned(bytes.to_vec()))
    }
}

impl<'a, T> heed::BytesDecode<'a> for ZeroCopyCodec<T> 
where
    T: for<'b> Serialize<DefaultSerializer<'b, AlignedVec, rkyv::rancor::Error>> + Archive + 'a
{
    type DItem = &'a T::Archived;
    
    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, heed::BoxedError> {
        unsafe { Ok(rkyv::access_unchecked::<Archived<T>>(bytes)) }
    }
}