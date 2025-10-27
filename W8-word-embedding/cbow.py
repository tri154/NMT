import re
import glob
from tqdm import tqdm


class IterableSentences:
    def __init__(self, dirname, lowercase=True, remove_punctuation=True):
        self.dirname = dirname
        self.lowercase = lowercase
        self.remove_punctuation =remove_punctuation


    def preprocessing(self,line:str):
        if self.lowercase:
            line = line.lower()
        if self.remove_punctuation:
            # Sử dụng re.sub để xóa các ký tự [\!"“”#$%&\*+,-./:;<=>?@^_`()|~=]|\n
            line = re.sub(pattern = r"[\!\"“”#$%&\*+,-./:;<=>?@^_`()|~=]|\n",
                             repl = " ",
                             string = line)
            line = re.sub(pattern = r"\s+",
                             repl = " ",
                             string = line)

        return line.strip(" ").split(" ")
    def __iter__(self):
        for fname in glob.glob(self.dirname+'/*.txt'):
            for line in open(fname):
                if len(self.preprocessing(line)) == 0:
                    continue
                else:
                    yield self.preprocessing(line)

    def __len__(self):
        return self.len

def get_ngram_list(sents, context_size=2):
    res = []
    for sent in sents:
        for i in range(context_size, len(sent) - context_size):
            context = [sent[i - j - 1] for j in range(CONTEXT_SIZE)] + [sent[i + 1 + j] for j in range(context_size)]
            target = sent[i]
            res.append((context, target))
    return res

from torch import nn, optim
from torch.nn import functional as F
import torch

class CBOW(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        hidden = self.embedding(inputs)
        hidden = F.tanh(hidden.mean(dim=0))
        log_probs = F.log_softmax(self.linear(hidden))
        return log_probs

def train(n_gram_list, word2id, id2word):
    EMBEDDING_DIM = 10
    losses = []
    loss_function = nn.NLLLoss()
    model = CBOW(len(vocab), CONTEXT_SIZE, EMBEDDING_DIM)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in tqdm(range(10)):
        total_loss = 0
        for context, target in n_gram_list[:100]:
            context_idxs = torch.tensor([word2id[w] for w in context], dtype=torch.long)

            model.zero_grad()

            log_probs = model(context_idxs)

            target_tensor = torch.zeros(len(vocab), dtype=torch.long)
            target_tensor[word2id[target]] = 1
            loss = loss_function(log_probs, target_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        print(total_loss)
    print(losses)

    print(model.embedding.weight[word2id["beauty"]])

if __name__ == "__main__":
    sents = IterableSentences('OpenCorpus')
    sents.len = 0

    vocab = set()
    for sent in sents:
        sents.len += 1
        for word in sent:
            vocab.add(word)
    # hyperparameters
    CONTEXT_SIZE=2
    ngram_list = get_ngram_list(sents, context_size=CONTEXT_SIZE)
    word2id = {word : i for i, word in enumerate(vocab)}
    id2word = {i: word for i, word in enumerate(vocab)}
    train(ngram_list, word2id, id2word)
