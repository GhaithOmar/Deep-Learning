import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from util import y2indicator, get_normalized_data


class HiddenLayerBatchNorm(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f

        W_init = np.random.randn(M1, M2) / np.sqrt(M1)
        self.W = theano.shared(W_init)
        self.gamma = theano.shared(np.ones(M2))
        self.beta = theano.shared(np.zeros(M2))
        self.rn_mean = theano.shared(np.zeros(M2))
        self.rn_var = theano.shared(np.zeros(M2))
        self.params = [self.W, self.gamma, self.beta]

    def forward(self, Z, is_training):
        a = Z.dot(self.W)
        if is_training:
            out, batch_mean, batch_invstd, new_rn_mean, new_rn_var = batch_normalization_train(
                a, self.gamma, self.beta, running_mean=self.rn_mean, running_var=self.rn_var)
            self.running_update = [
                (self.rn_mean, new_rn_mean),
                (self.rn_var, new_rn_var)
            ]
        else:
            out = batch_normalization_test(
                a, self.gamma, self.beta, self.rn_mean, self.rn_var)
        return self.f(out)


class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f

        W_init = np.random.randn(M1, M2) / np.sqrt(M1)
        b_init = np.random.randn(M2)

        self.W = theano.shared(W_init)
        self.b = theano.shared(b_init)

        self.params = [self.W, self.b]

    def forward(self, Z):
        return self.f(Z.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sz):
        self.hidden_layer_sz = hidden_layer_sz

    def fit(
            self,
            Xtrain,
            Xtest,
            Ytrain,
            Ytest,
            activation=T.nnet.relu,
            reg=0.01,
            lr=1e-2,
            batch_sz=1,
            epochs=20,
            mu=0.9,
            show_fig=False):
        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytrain = Ytrain.astype(np.int32)
        Ytest = Ytest.astype(np.int32)
        N, D = Xtrain.shape
        K = len(set(Ytrain))
        n_batches = N // batch_sz

        self.hidden_layer = []
        M1 = D
        for M2 in self.hidden_layer_sz:
            h = HiddenLayerBatchNorm(M1, M2, activation)
            self.hidden_layer.append(h)
            M1 = M2

        self.hidden_layer.append(HiddenLayer(M1, K, T.nnet.softmax))
        self.params = []
        for h in self.hidden_layer:
            self.params += h.params

        thX = T.matrix('X')
        thT = T.ivector('T')
        self.thX = thX

        thY = self.forward(thX, is_training=True)

        dparams = [theano.shared(np.zeros_like(p.get_value()))
                   for p in self.params]

        cost = -T.mean(T.log(thY[T.arange(thT.shape[0]), thT]))

        grads = T.grad(cost, self.params)

        self.y_pred = self.forward(thX, is_training=False)

        self.prediction = T.argmax(self.y_pred, axis=1)

        updates = [(p,
                    p + mu * dp - lr * g) for p,
                   dp,
                   g in zip(self.params,
                            dparams,
                            grads)] + [(dp,
                                        mu * dp - lr * g) for dp,
                                       g in zip(dparams,
                                                grads)]

        for layer in self.hidden_layer[:-1]:
            updates += layer.running_update

        train_op = theano.function(
            inputs=[thX, thT],
            updates=updates
        )

        cost_predict_op = theano.function(
            inputs=[thX, thT],
            outputs=[cost]
        )
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[self.prediction]
        )

        ll = []
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Ytrain[j * batch_sz:(j * batch_sz + batch_sz)]

                train_op(Xbatch, Ybatch)
                if j % 100 == 0:
                    cost_val = cost_predict_op(Xtest, Ytest)
                    ll.append(cost_val)
                    acc = self.score(Xtest, Ytest)
                    self.acc = acc
                    print(
                        f'in epoch number {i} in batch number{j}  cost = {cost_val}  acc= {acc}')

        print(f'Final acc= {self.acc}')
        if show_fig:
            plt.plot(ll)
            plt.title('Cost')
            plt.show()

    def forward(self, X, is_training):
        Z = X
        for h in self.hidden_layer[:-1]:
            Z = h.forward(Z, is_training)

        return self.hidden_layer[-1].forward(Z)

    def score(self, X, Y):
        pY = self.predict(X)
        return np.mean(pY == Y)

    def predict(self, X):
        return self.predict_op(X)


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    ann = ANN([500, 100])
    ann.fit(Xtrain, Xtest, Ytrain, Ytest, batch_sz=500)
