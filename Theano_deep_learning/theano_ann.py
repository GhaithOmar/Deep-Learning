import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from util import get_normalized_data


class HiddenLayer(object):
    def __init__(self, M1, M2, id_n):
        self.id_n = id_n
        self.M1 = M1
        self.M2 = M2

        W = np.random.randn(M1, M2) / np.sqrt(M1)
        b = np.zeros(M2)

        self.W = theano.shared(W, f'W_{self.id_n}')
        self.b = theano.shared(b, f'b_{self.id_n}')

        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_sz):
        self.hidden_sz = hidden_sz

    def fit(
            self,
            X,
            Y,
            Xvalid=None,
            Yvalid=None,
            lr=1e-4,
            epochs=20,
            batch_sz=1,
            reg=0.,
            mu=0.9,
            show_fig=False):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int32)
        N, D = X.shape
        K = len(set(Y))

        M1 = D
        count = 0
        self.hidden_layer = []
        for M2 in self.hidden_sz:
            h = HiddenLayer(M1, M2, count)
            M1 = M2
            count += 1
            self.hidden_layer.append(h)

        w_init = np.random.randn(M1, K) / np.sqrt(M1)
        b_init = np.zeros(K)
        self.W = theano.shared(w_init, 'W_f')
        self.b = theano.shared(b_init, 'b_f')

        self.params = [self.W, self.b]

        for h in self.hidden_layer:
            self.params += h.params

        # Create equations

        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        reg_cost = reg * T.sum([(p * p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + reg_cost

        grads = T.grad(cost, self.params)
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]

        prediction = T.argmax(pY, axis=1)

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        cost_prediction_eq = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction]
        )

        # batch gradient descent
        n_batches = N // batch_sz
        ll = []
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_prediction_eq(Xvalid, Yvalid)
                    ll.append(c)
                    acc = self.score(p, Yvalid)
                    print(f'now epoch {i} batch {j}  cost: {c}  acc: {acc} ')

        if show_fig:
            plt.plot(ll)
            plt.title('cost')
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layer:
            Z = h.forward(Z)

        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def score(self, pY, T):
        return np.mean(pY == T)


def main():
    #X, Y = getData()
    X_trian, Xtest, Y_train, Ytest = get_normalized_data()

    model = ANN([200, 100])
    model.fit(X_trian, Y_train, Xtest, Ytest, batch_sz=500,show_fig=True)


if __name__ == '__main__':
    main()
