import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def load_data(tag):
    cur_dir = os.getcwd()
    tag_dir = os.path.join(cur_dir, "aclImdb", tag)
    pos_set = os.path.join(tag_dir, 'pos')
    neg_set = os.path.join(tag_dir, 'neg')

    positive_label = 1
    negative_label = 0
    positive = load_set(pos_set, label=positive_label)
    negative = load_set(neg_set, label=negative_label)

    res = positive + negative
    return res

def load_set(path, label):
    files = os.listdir(path)
    res = list()
    for file in files:
        match = re.match(r"(\d+)_(\d+)\.txt$", file)
        num1, num2 = match.groups()
        num1, num2 = int(num1), int(num2)
        rating = num2
        id = num1
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            data = f.read()
        label = label

        point = {
            "id": id,
            "data": data,
            "label": label,
            "rating": rating,
        }
        res.append(point)
    return res

def load_vocab():
    cur_dir = os.getcwd()
    vocab_path = os.path.join(cur_dir, "aclImdb/imdb.vocab")
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.split()
    return data

def prepare_data(dataset, vectorizer):
    texts = [item["data"] for item in dataset]
    labels = [item["label"] for item in dataset]
    X = vectorizer.transform(texts)
    y = np.array(labels)
    return X, y

def prepare_data_torch(dataset, vectorizer):
    texts = [item["data"] for item in dataset]
    labels = [item["label"] for item in dataset]
    X = torch.tensor(vectorizer.transform(texts).toarray())
    y = torch.tensor(labels)
    return X, y

def naive_bayes():
    vocab = load_vocab()
    train_set = load_data('train')
    test_set = load_data('test')
    vectorizer = CountVectorizer(vocabulary=vocab)
    vectorizer.fit([item["data"] for item in train_set])

    x_train, y_train = prepare_data(train_set, vectorizer)
    x_test, y_test = prepare_data(test_set, vectorizer)

    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")  # since labels are 0/1

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

def svm_bow():
    vocab = load_vocab()
    train_set = load_data('train')
    test_set = load_data('test')
    vectorizer = CountVectorizer(vocabulary=vocab)
    vectorizer.fit([item["data"] for item in train_set])

    x_train, y_train = prepare_data(train_set, vectorizer)
    x_test, y_test = prepare_data(test_set, vectorizer)

    clf = LinearSVC(random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")  # since labels are 0/1

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

def svm_tfidf():
    vocab = load_vocab()
    train_set = load_data("train")
    test_set = load_data("test")

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    vectorizer.fit([item["data"] for item in train_set])

    X_train, y_train = prepare_data(train_set, vectorizer)
    X_test, y_test = prepare_data(test_set, vectorizer)

    clf = LinearSVC(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")


class IMDBDataset(Dataset):
    def __init__(self, text_embs, labels):
        self.X = text_embs
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {acc:.4f}")

def mlp_tfidf():
    vocab = load_vocab()
    train_set = load_data("train")
    test_set = load_data("test")
    vectorizer = TfidfVectorizer(vocabulary=vocab, max_features=1000)
    vectorizer.fit([item["data"] for item in train_set])

    x_train, y_train = prepare_data_torch(train_set, vectorizer)
    x_test, y_test = prepare_data_torch(test_set, vectorizer)

    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_size = x_train.shape[-1]
    hidden_dim = 256
    output_size = 2
    model = MLP(input_size, hidden_dim, output_size)

    train_model(model, train_loader, test_loader, epochs=5, lr=1e-3, device="cpu")

if __name__ == "__main__":
    mlp_tfidf()
