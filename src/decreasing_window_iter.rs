use std::{iter::FusedIterator, num::NonZero};

pub struct DecreasingWindows<'a, T: 'a> {
    v: &'a [T],
    size: NonZero<usize>,
}
impl<'a, T: 'a> DecreasingWindows<'a, T> {
    #[inline]
    pub fn new(slice: &'a [T], size: NonZero<usize>) -> Self {
        Self { v: slice, size }
    }
}
impl<'a, T> Iterator for DecreasingWindows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size.get() > self.v.len() {
            self.size = NonZero::new(self.v.len())?;
        }

        let ret = Some(&self.v[..self.size.get()]);
        self.v = &self.v[1..];
        ret
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.v.len();
        (size, Some(size))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}
impl<T> ExactSizeIterator for DecreasingWindows<'_, T> {}
impl<T> FusedIterator for DecreasingWindows<'_, T> {}
