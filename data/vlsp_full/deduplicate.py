from datasketch import MinHash, MinHashLSH
import re
from tqdm import tqdm

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_ngrams(text, n=2):
    tokens = text.split()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def build_minhash(shingles, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m


def deduplicate_keep_longest(
    src_sentences,
    tgt_sentences,
    jaccard_threshold=0.85,
    num_perm=128,
    ngram=2
):
    assert len(src_sentences) == len(tgt_sentences)

    lsh = MinHashLSH(
        threshold=jaccard_threshold,
        num_perm=num_perm
    )

    minhashes = {}
    keep = [True] * len(src_sentences)

    for i, src in enumerate(src_sentences):
        src_norm = normalize(src)
        shingles = word_ngrams(src_norm, n=ngram)

        if not shingles:
            continue

        m = build_minhash(shingles, num_perm)
        candidates = lsh.query(m)

        if not candidates:
            # No duplicate found
            lsh.insert(i, m)
            minhashes[i] = m
            continue

        # Compare with existing near-duplicates
        replace = False
        for j in candidates:
            if len(src_sentences[i]) > len(src_sentences[j]):
                # New sentence is longer → replace old
                keep[j] = False
                lsh.remove(j)
                minhashes.pop(j, None)
                replace = True
            else:
                # Existing one is longer → discard current
                keep[i] = False
                break

        if replace and keep[i]:
            lsh.insert(i, m)
            minhashes[i] = m
        else:
            keep[i] = False

    dedup_src = [s for i, s in enumerate(src_sentences) if keep[i]]
    dedup_tgt = [t for i, t in enumerate(tgt_sentences) if keep[i]]

    return dedup_src, dedup_tgt



def deduplicate_parallel(
    src_sentences,
    tgt_sentences,
    jaccard_threshold=0.85,
    num_perm=128,
    ngram=2
):
    assert len(src_sentences) == len(tgt_sentences)

    lsh = MinHashLSH(
        threshold=jaccard_threshold,
        num_perm=num_perm
    )

    minhashes = []
    keep = [True] * len(src_sentences)

    for i, src in tqdm(enumerate(src_sentences)):
        src_norm = normalize(src)
        shingles = word_ngrams(src_norm, n=ngram)

        if not shingles:
            continue

        m = build_minhash(shingles, num_perm)
        minhashes.append(m)

        # Query for near duplicates
        candidates = lsh.query(m)

        if candidates:
            # Found near-duplicate → discard this one
            keep[i] = False
        else:
            # No duplicate → insert into index
            lsh.insert(i, m)

    dedup_src = [s for i, s in enumerate(src_sentences) if keep[i]]
    dedup_tgt = [t for i, t in enumerate(tgt_sentences) if keep[i]]

    return dedup_src, dedup_tgt


if __name__ == "__main__":
    src_sents = open("train.en", "r").readlines()
    trg_sents = open("train.vi", "r").readlines()

    # src_dedup, tgt_dedup = deduplicate_keep_longest(src_sents, trg_sents)
    src_dedup, tgt_dedup = deduplicate_parallel(src_sents, trg_sents)

    print(len(src_sents))
    print(len(src_dedup))
    breakpoint()
    open("out.src","w").write("".join(src_dedup))
    open("out.tgt","w").write("".join(tgt_dedup))
