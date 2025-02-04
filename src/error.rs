use thiserror::Error;

/// Possible errors that can occur while interacting with the database.
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Lmdb error: {0}")]
    LmdbError(#[from] heed::Error),

    #[error("Serialize error: {0}")]
    EncodingError(#[from] rkyv::rancor::Error),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Key `{0}` not found in database `{1}`")]
    KeyNotFound(String, String),
}

/// Possible errors that can occur while searching.
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Db error: {0}")]
    DatabaseError(#[from] DatabaseError),

    #[error("Searched query is empty")]
    EmptyQuery,

    #[error("No combination found while trying to merge and minimize")]
    MergeAndMinimizeNotPossible,

    #[error("Token `{0}` not found in the database")]
    TokenNotFound(String),

    #[error("Empty Intersection")]
    EmptyIntersection,

    #[error("Catastrophic error has occurred")]
    InternalError,
}

/// Possible errors when trying to retrieve documents by their internal ID.
#[derive(Error, Debug)]
pub enum DocumentError {
    #[error("Db error: {0}")]
    DatabaseError(#[from] DatabaseError),

    #[error("Document with id `{0}` not found")]
    DocumentNotFound(u32),
}
