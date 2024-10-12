use std::{alloc::{alloc, dealloc, Allocator}, ptr::NonNull};

use rkyv::{Archive, Serialize};

#[derive(Default, Archive, Serialize)]
pub struct AlignedAllocator<const N: usize>;
unsafe impl<const N: usize> Allocator for AlignedAllocator<N> {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        unsafe {
            let p = alloc(layout.align_to(N).unwrap());
            let s = std::ptr::slice_from_raw_parts_mut(p, layout.size());
            #[cfg(debug_assertions)]
            return NonNull::new(s).ok_or(std::alloc::AllocError);

            #[cfg(not(debug_assertions))]
            return Ok(NonNull::new_unchecked(s));
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        dealloc(ptr.as_ptr(), layout.align_to(N).unwrap());
    }
}

pub type Aligned64 = AlignedAllocator<64>;