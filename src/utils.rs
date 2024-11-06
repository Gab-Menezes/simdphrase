use unicode_segmentation::UnicodeSegmentation;

pub fn normalize(s: &str) -> String {
    s.trim_start().trim_end().to_lowercase()
}

pub fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|t| {
        if !t.is_empty() {
            return !t.chars().next().unwrap().is_whitespace();
        }
        false
    })
}
