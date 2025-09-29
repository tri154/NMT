import numpy as np

class LogisticRegression:
    def __init__(self, dim_input):
        self.w = np.zeros(dim_input, dtype=float)
        self.b = 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)

    def predict(self, x, db=0.5):
        out = self.forward(x)
        out = np.where(out > db, 1, 0)
        return out

    def update(self, g_w, g_b, lr):
        self.w = self.w + lr * g_w
        self.b = self.b + lr * g_b

def loss_fnt(out, labels):
    n_sample = out.shape[0]
    res = - labels * np.log(out) - (1.0 - labels) * np.log(1.0 - out)
    return 1/n_sample * np.sum(res)

if __name__ == "__main__":
    np.random.seed(42)
    n_sample, dim_input = 200, 2
    lr = 0.1

    X_class0 = np.random.randn(n_sample//2, 2) + np.array([-2, -2])
    X_class1 = np.random.randn(n_sample//2, 2) + np.array([2, 2])

    X = np.vstack([X_class0, X_class1])
    labels = np.array([0]*(n_sample//2) + [1]*(n_sample//2))

    model = LogisticRegression(dim_input)
    n_epochs = 100
    for epid in range(n_epochs):
        out = model.forward(X)
        loss = loss_fnt(out, labels)
        print(loss)

        b_w = (labels - out)
        b_w = 1/n_sample * np.sum(b_w)

        g_w = (labels - out).reshape(-1, 1) * X
        g_w = 1/n_sample * np.sum(g_w, axis=0)

        model.update(g_w, b_w, lr)

    predictions = model.predict(X)

    acc = np.count_nonzero(predictions == labels) / n_sample
    print("accuracy: ", acc)
