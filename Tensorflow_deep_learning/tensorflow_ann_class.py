import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import get_normalized_data


class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.f = f
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) / np.sqrt(M1)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        self.params = [self.W, self.b]

    def forward(self, X):
        #print(X.shape)
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hiddden_layers_sz):
        self.hiddden_layers_sz = hiddden_layers_sz

    def fit(
            self,
            X,
            Y,
            Xvalid,
            Yvalid,
            activation=tf.nn.relu,
            lr=1e-3,
            batch_sz=1,
            epochs=20,
            show_fig=False):
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int64)

        N, D = X.shape
        K = len(set(Y))

        n_batches = N // batch_sz

        self.hidden_layer = []
        M1 = D
        for M2 in self.hiddden_layers_sz:
            h = HiddenLayer(M1, M2, activation)
            self.hidden_layer.append(h)
            M1 = M2

        w_init = np.random.randn(M1, K) / np.sqrt(M1)
        b_init = np.zeros(K)

        self.W = tf.Variable(w_init.astype(np.float32))
        self.b = tf.Variable(b_init.astype(np.float32))

        self.params = [self.W, self.b]

        for h in self.hidden_layer:
            self.params.extend(h.params)

        thX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        T = tf.placeholder(tf.int64, shape=(None,), name='T')

        yish = self.forward(thX)
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=yish,
                labels=T
            )
        )

        prediction = tf.argmax(yish, 1)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        init = tf.global_variables_initializer()

        ll = []

        with tf.Session() as s:
            s.run(init)
            for i in range(epochs):
                for j in range(n_batches):
                    Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                    Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]

                    s.run(train_op, feed_dict={thX: Xbatch, T: Ybatch})
                    # we pass the RUN

                    if j % 50 == 0:
                        c = s.run(cost, feed_dict={thX: Xvalid, T: Yvalid})

                        pY = s.run(
                            prediction, feed_dict={
                                thX: Xvalid, T: Yvalid})
                        ll.append(c)
                        acc = self.score(pY, Yvalid)

                        print(
                            f'epoch number {i} batch {j}  cost= {c}    , acc= {acc}')

        if show_fig:
            plt.plot(ll)
            plt.title('cost')
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layer:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def score(self, pY, T):
        return np.mean(pY == T)


if __name__ == '__main__':

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    ann = ANN([200, 100, 50])
    ann.fit(Xtrain, Ytrain, Xtest, Ytest, batch_sz=500, show_fig=True)
