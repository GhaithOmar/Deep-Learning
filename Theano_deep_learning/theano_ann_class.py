import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from util import y2indicator, get_normalized_data


class HiddenLayer(object):
    def __init__(self, M1, M2, n):
        self.M1 = M1
        self.M2 = M2
        self.n = n

        W_init = np.random.randn(M1, M2) / np.sqrt(M1)
        b_init = np.random.randn(M2)

        self.W = theano.shared(W_init, f'W{self.n}')
        self.b = theano.shared(b_init, f'b{self.n}')

        self.params = [self.W, self.b]

    def forward(self, Z):
        return T.nnet.relu(Z.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sz):
        self.hidden_layer_sz = hidden_layer_sz

    def fit(
            self,
            Xtrain,
            Xtest,
            Ytrain,
            Ytest,
            reg=0.01,
            lr=1e-4,
            batch_sz=1,
            epochs=20):
        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytrain = Ytrain.astype(np.int32)
        Ytest = Ytest.astype(np.int32)
        N, D = Xtrain.shape
        K = len(set(Ytrain))
        n_batches = N // batch_sz

        self.hidden_layer = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sz:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layer.append(h)
            M1 = M2
            count += 1

        w_init = np.random.randn(M1, K) / np.sqrt(M1)
        b_init = np.zeros(K)

        self.W = theano.shared(w_init, f'W{count}')
        self.b = theano.shared(b_init, f'b{count}')

        self.params = [self.W, self.b]
        for h in self.hidden_layer:
            self.params += h.params

        thX = T.matrix('X')
        thT = T.ivector('T')

        thY = self.forward(thX)

        dparams = [theano.shared(np.zeros_like(p.get_value()))
                   for p in self.params]

        cost = -T.mean(T.log(thY[T.arange(thT.shape[0]), thT]))

        grads = T.grad(cost, self.params)

        prediction = T.argmax(thY, axis=1)
        mu = 0.9

        updates = [(p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)]
        updates +=[(dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)]

        train_op = theano.function(
            inputs=[thX, thT],
            updates=updates
        )

        cost_predict_op = theano.function(
            inputs=[thX, thT],
            outputs=[cost, prediction]
        )

        ll = []
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Ytrain[j * batch_sz:(j * batch_sz + batch_sz)]

                train_op(Xbatch, Ybatch)
                if j % 100 == 0:
                    cost_val, predict_val = cost_predict_op(Xtest, Ytest)
                    ll.append(cost_val)
                    acc = self.score(predict_val, Ytest)
                    self.acc = acc
                    print(
                        f'in epoch number {i} in batch number{j}  cost = {cost_val}  acc= {acc}')

        print(f'Final acc= {self.acc}')
        if show_fig:
            plt.plot(ll)
            plt.title('Cost')
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layer:
            Z = h.forward(Z)

        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def score(self, pY, Y):
        return np.mean(pY == Y)


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    ann = ANN([500, 100])
    ann.fit(Xtrain, Xtest, Ytrain, Ytest, batch_sz=500)
