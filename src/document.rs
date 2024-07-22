use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Serialize, Deserialize, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[archive_attr(derive(Debug))]
pub struct Document {
    pub content: Option<String>,
}