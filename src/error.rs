use thiserror::Error;

/// Possible errors that can occur while interacting with the database.
#[derive(Error, Debug)]
pub enum DbError {
    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Lmdb error: {0}")]
    LmdbError(#[from] heed::Error),

    #[error("Serialize error: {0}")]
    EncodingError(#[from] rkyv::rancor::Error),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Key `{0}` not found in database `{1}`")]
    KeyNotFound(String, String)
}

/// Possible errors that can occur while searching.
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Db error: {0}")]
    DbError(#[from] DbError),

    #[error("Searched query is empty")]
    EmptyQuery,

    #[error("No combination found while trying to merge and minimize")]
    MergeAndMinimizeNotPossible,

    #[error("Token `{0}` not found in the database")]
    TokenNotFound(String),

    #[error("Empty Intersection")]
    EmptyIntersection,

    #[error("Catastrophic error has occurred")]
    InternalError
}