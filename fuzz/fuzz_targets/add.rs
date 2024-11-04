#![feature(allocator_api)]

#![no_main]

use libfuzzer_sys::fuzz_target;
use phrase_search::AlignedRoaringishPacked;
use phrase_search::AlignedBorrowRoaringishPacked;
use phrase_search::Aligned64;

fuzz_target!(|r: (AlignedRoaringishPacked, u32)| {
    let (lhs, len) = r;
    let lhs = AlignedBorrowRoaringishPacked::new(&lhs);
    let len = (len % 15) + 1;

    let r0 = &lhs + len;
    let r1 = &lhs + (len as u16);

    assert_eq!(r0, r1);
});
