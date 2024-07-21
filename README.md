# Phrase Search POC

## Index
`cargo run --release -- index-folder <FOLDER> <INDEX_NAME> <FILES_PER_INDEX> <DB_SIZE> <MAX_DOC_ID>`

Example: `cargo run --release -- index-folder ./data ./db 80 2147483648 5000000`

Note: Add `--recursive` at the end if the `.jsonl` files are in folders inside the main folder

* During indexing the data files will be broken in `FILES_PER_INDEX` chunks, each of this chuck will become a shard
* Common tokens will be merged together
* Some information about the indexing will be tracked, since Heed leaks a shit ton of memory we use this tracked data to continue the indexing if it dies. To continue the indexing just run the same command.

## Search
`sudo ./target/release/phrase_search search <RUNS> <QUERIES> <INDEX_NAME> <DB_SIZE>`

Example: `sudo ./target/release/phrase_search search 100 ./queries.txt ./db 2147483648`