import numpy as np
src_sents = open("train_dedup.en", "r").readlines()
trg_sents = open("train_dedup.vi", "r").readlines()
assert len(src_sents) == len(trg_sents)
lengths = [3000, 15000, len(src_sents) - 3000 - 15000]

idx = np.random.permutation(len(src_sents))

src_sents = [src_sents[i] for i in idx]
trg_sents = [trg_sents[i] for i in idx]

s1, s2 = lengths[0], lengths[0] + lengths[1]

src_split = [
    src_sents[:s1],
    src_sents[s1:s2],
    src_sents[s2:]
]

trg_split = [
    trg_sents[:s1],
    trg_sents[s1:s2],
    trg_sents[s2:]
]

def save(path, idx):
    with open(f"{path}.en", "w") as f:
        f.writelines(src_split[idx])
    with open(f"{path}.vi", "w") as f:
        f.writelines(trg_split[idx])

save("dev", 0)
save("gpro", 1)
save("train_splited", 2)
