"""Export the cloud catalog index using only local reference audio."""

import numpy as np
from fingerprint import fingerprints, spectrum
from loudio_catalog import load_loudio_references
from loudio_fingerprints import ASSETS, FORMAT, INDEX_FILE, token_digest


def main():
    references, titles = load_loudio_references()
    keys = sorted(references, key=lambda key: (titles[key].casefold(), key))
    digest_chunks, posting_chunks = [], []
    for idx, key in enumerate(keys):
        pairs = list(fingerprints(np.abs(spectrum(references[key])[2])))
        digest_chunks.append(
            np.fromiter((token_digest(token) for token, _ in pairs), dtype=np.uint64)
        )
        posting_chunks.append(np.array([(idx, frame) for _, frame in pairs], dtype=np.uint16))

    digests = np.concatenate(digest_chunks)
    order = np.argsort(digests, kind="stable")
    unique, starts = np.unique(digests[order], return_index=True)
    postings = np.concatenate(posting_chunks)[order]

    ASSETS.mkdir(exist_ok=True)
    np.savez_compressed(
        INDEX_FILE,
        format=FORMAT,
        titles=np.array([titles[key] for key in keys]),
        digests=unique,
        starts=np.append(starts, len(postings)).astype(np.uint32),
        postings=postings,
    )
    print(f"{len(keys)} recordings, {len(unique):,} keys, {len(postings):,} postings")


if __name__ == "__main__":
    main()
