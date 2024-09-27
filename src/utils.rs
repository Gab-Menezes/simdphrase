use unicode_segmentation::UnicodeSegmentation;

pub const MAX_SEQ_LEN: usize = 5;

pub fn normalize(s: &str) -> String {
    s.trim_start().trim_end().to_lowercase()
}

pub fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|t| {
        if t.len() >= 1 {
            let t = t.chars().next().unwrap();
            return t > ' ';
        }
        return false;
    })
}
