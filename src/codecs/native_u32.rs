use std::borrow::Cow;

use heed::BoxedError;

pub struct NativeU32;

impl<'a> heed::BytesDecode<'a> for NativeU32 {
    type DItem = u32;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        unsafe { Ok(u32::from_ne_bytes(bytes.try_into().unwrap_unchecked())) }
    }
}

impl<'a> heed::BytesEncode<'a> for NativeU32 {
    type EItem = u32;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let p = item as *const u32 as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(p, std::mem::size_of::<Self::EItem>()) };
        Ok(Cow::Borrowed(bytes))
    }
}
