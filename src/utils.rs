use unicode_segmentation::UnicodeSegmentation;

/// Normalizes the input string by trimming leading and trailing
/// whitespaces and converting it to lowercase.
pub fn normalize(s: &str) -> String {
    s.trim_start().trim_end().to_lowercase()
}

/// Tokenizes the input string by splitting it into word bounds
/// also remove all tokens that are considered whitespace by utf-8.
pub fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|t| {
        if !t.is_empty() {
            // This is safe because we know that `t` is not empty.
            return !t.chars().next().unwrap().is_whitespace();
        }
        false
    })
}
