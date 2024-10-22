use unicode_segmentation::UnicodeSegmentation;

pub fn normalize(s: &str) -> String {
    s.trim_start().trim_end().to_lowercase()
}

pub fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|t| {
        if !t.is_empty() {
            let t = t.chars().next().unwrap();
            return t > ' ';
        }
        false
    })
}
