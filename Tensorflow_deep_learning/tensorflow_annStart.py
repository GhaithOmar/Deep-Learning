import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import get_normalized_data


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    Ytrain = Ytrain.astype(np.int32)
    Ytest = Ytest.astype(np.int32)
    batch_sz = 50

    N, D = Xtrain.shape
    M = 300
    K = len(set(Ytrain))

    n_batches = N // batch_sz

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    Y = tf.placeholder(tf.int32, shape=(None,), name='Y')

    W1_init = (np.random.randn(D, M) / np.sqrt(D)).astype(np.float32)
    b1_init = np.zeros(M).astype(np.float32)
    W2_init = (np.random.randn(M, K) / np.sqrt(M)).astype(np.float32)
    b2_init = np.zeros(K).astype(np.float32)

    W1 = tf.Variable(W1_init)
    b1 = tf.Variable(b1_init)
    W2 = tf.Variable(W2_init)
    b2 = tf.Variable(b2_init)

    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    Yish = tf.matmul(Z, W2) + b2
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=Yish,
            labels=Y
        )
    )
    #train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
    predict_op = tf.argmax(Yish, 1)
    # call global variables initializer

    init = tf.global_variables_initializer()
    ll = []
    with tf.Session() as session:
        session.run(init)

        for i in range(20):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Ytrain[j * batch_sz:(j * batch_sz + batch_sz)]

                session.run(train_op, feed_dict={X: Xbatch, Y: Ybatch})

                if j % 50 == 0:
                    test_cost = session.run(
                        cost, feed_dict={X: Xtest, Y: Ytest})
                    pY = session.run(predict_op, feed_dict={X: Xtest})
                    acc = score(pY, Ytest)
                    ll.append(test_cost)

                    print(
                        f'at epoch {i} batch {j} cost= {test_cost}  acc={acc}')

    plt.plot(ll)
    plt.show()


def score(pY, Y):
    return np.mean(pY == Y)


if __name__ == '__main__':
    main()
