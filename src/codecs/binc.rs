use std::{borrow::Cow, marker::PhantomData};

use bincode::{
    config::{FixintEncoding, WithOtherEndian, WithOtherIntEncoding},
    DefaultOptions, Options,
};
use heed::BoxedError;
use serde::{Deserialize, Serialize};

pub type LittleEndianVariableOption = WithOtherEndian<DefaultOptions, bincode::config::LittleEndian>;
pub type LittleEndianFixOption = WithOtherIntEncoding<
    WithOtherEndian<DefaultOptions, bincode::config::LittleEndian>,
    FixintEncoding,
>;

pub trait NewBincodeOptions {
    fn new() -> Self;
}

impl NewBincodeOptions for LittleEndianVariableOption {
    fn new() -> Self {
        bincode::DefaultOptions::new().with_little_endian()
    }
}

impl NewBincodeOptions for LittleEndianFixOption {
    fn new() -> Self {
        bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
    }
}

pub struct BincodeCodec<T, O>(PhantomData<(T, O)>)
where
    T: Serialize,
    T: for<'a> Deserialize<'a>,
    O: Options + NewBincodeOptions;

impl<'b, T, O> heed::BytesEncode<'b> for BincodeCodec<T, O>
where
    T: Serialize + 'b,
    T: for<'a> Deserialize<'a>,
    O: Options + NewBincodeOptions,
{
    type EItem = T;

    fn bytes_encode(item: &'_ Self::EItem) -> Result<Cow<'_, [u8]>, BoxedError> {
        let bytes = O::new().serialize(item)?;
        Ok(Cow::Owned(bytes))
    }
}

impl<'b, T, O> heed::BytesDecode<'b> for BincodeCodec<T, O>
where
    T: Serialize + 'b,
    T: for<'a> Deserialize<'a>,
    O: Options + NewBincodeOptions,
{
    type DItem = T;

    fn bytes_decode(bytes: &'_ [u8]) -> Result<Self::DItem, BoxedError> {
        let item = O::new().deserialize(bytes)?;
        Ok(item)
    }
}