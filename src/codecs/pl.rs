use heed::BoxedError;

use crate::pl::UnsafePostingList;

pub struct UnsafePostingListCodec;

impl<'a> heed::BytesDecode<'a> for UnsafePostingListCodec {
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