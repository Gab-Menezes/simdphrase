use std::{borrow::Cow, marker::PhantomData};

use bincode::{config::WithOtherEndian, DefaultOptions, Options};
use heed::BoxedError;
use serde::{Deserialize, Serialize};

pub type BigEndianVariableOption = WithOtherEndian<DefaultOptions, bincode::config::BigEndian>;

pub trait CreateBincodeOptions {
    fn create() -> Self;
}

impl CreateBincodeOptions for BigEndianVariableOption {
    fn create() -> Self {
        bincode::DefaultOptions::new().with_big_endian()
    }
}

pub struct BincodeCodec<T, O>(PhantomData<(T, O)>)
where
    T: Serialize,
    T: for<'a> Deserialize<'a>,
    O: Options + CreateBincodeOptions;

impl<'b, T, O> heed::BytesEncode<'b> for BincodeCodec<T, O>
where
    T: Serialize + 'b,
    T: for<'a> Deserialize<'a>,
    O: Options + CreateBincodeOptions,
{
    type EItem = T;

    fn bytes_encode(item: &'_ Self::EItem) -> Result<Cow<'_, [u8]>, BoxedError> {
        let bytes = O::create().serialize(item)?;
        Ok(Cow::Owned(bytes))
    }
}

impl<'b, T, O> heed::BytesDecode<'b> for BincodeCodec<T, O>
where
    T: Serialize + 'b,
    T: for<'a> Deserialize<'a>,
    O: Options + CreateBincodeOptions,
{
    type DItem = T;

    fn bytes_decode(bytes: &'_ [u8]) -> Result<Self::DItem, BoxedError> {
        let item = O::create().deserialize(bytes)?;
        Ok(item)
    }
}