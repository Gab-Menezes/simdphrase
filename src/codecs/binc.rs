use std::{borrow::Cow, marker::PhantomData};

use bincode::{
    config::{FixintEncoding, WithOtherEndian, WithOtherIntEncoding},
    DefaultOptions, Options,
};
use heed::BoxedError;
use serde::{Deserialize, Serialize};

use crate::pl::UnsafePostingList;

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

pub struct UnsafeBigEndianFixOptionCodec;

impl<'a> heed::BytesDecode<'a> for UnsafeBigEndianFixOptionCodec {
    type DItem = UnsafePostingList<'a>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        const S_U32: usize = std::mem::size_of::<u32>();
        const S_USIZE: usize = std::mem::size_of::<usize>();
        unsafe {
            let l = (bytes.get_unchecked(0..S_USIZE).as_ptr() as *const usize).read_unaligned();
            let doc_ids = std::slice::from_raw_parts(bytes.as_ptr().add(S_USIZE) as *const u32, l);
    
            let mut b = S_USIZE + (l * S_U32);
    
            let l = (bytes.get_unchecked(b..(b + S_USIZE)).as_ptr() as *const usize).read_unaligned();
            let mut positions: Box<[std::mem::MaybeUninit<&[u32]>]> = Box::new_uninit_slice(l);
            b += S_USIZE;
            for i in 0..l {
                let e = b + S_USIZE;
                let l = (bytes.get_unchecked(b..e).as_ptr() as *const usize).read_unaligned();
                let s = std::slice::from_raw_parts(bytes.as_ptr().add(e) as *const u32, l);
                b = e + (l * S_U32);
                positions[i].as_mut_ptr().write(s);
            }
            Ok(UnsafePostingList {
                doc_ids,
                positions: positions.assume_init(),
            })
        }
    }
}
