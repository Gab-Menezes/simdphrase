use std::{borrow::Cow, marker::PhantomData};

use rkyv::{ser::{serializers::AllocSerializer, Serializer}, Archive, Fallible, Serialize};

pub struct ZeroCopyCodec<T, const N: usize>(PhantomData<T>)
where
    T: Serialize<AllocSerializer<N>>;

impl<'a, T, const N: usize> heed::BytesEncode<'a> for ZeroCopyCodec<T, N> 
where
    T: Serialize<AllocSerializer<N>> + 'a
{
    type EItem = T;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, heed::BoxedError> {
        let bytes = rkyv::to_bytes::<T, N>(item).unwrap();
        Ok(Cow::Owned(bytes.to_vec()))
    }
}

impl<'a, T, const N: usize> heed::BytesDecode<'a> for ZeroCopyCodec<T, N> 
where
    T: Serialize<AllocSerializer<N>> + 'a
{
    type DItem = &'a T::Archived;
    
    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, heed::BoxedError> {
        Ok(unsafe { rkyv::archived_root::<T>(bytes) })
    }
}