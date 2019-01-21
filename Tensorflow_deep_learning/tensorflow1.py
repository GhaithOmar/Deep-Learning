import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    max_iter = 20
    print_period = 10
    X_train, X_test, Y_train, Y_test = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Y_train_ind = y2indicator(Y_train)
    Y_test_ind = y2indicator(Y_test)

    N, D = X_train.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M1 = 300
    M2 = 100
    K = 10

    W1_init = np.random.randn(D, M1) / 28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(K)
    b3_init = np.zeros(K)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Yish = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=Yish, labels=T))
    train_op = tf.train.RMSPropOptimizer(
        lr, decay=0.99, momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish, 1)
    init = tf.global_variables_initializer()
    ll = []
    with tf.Session() as s:
        s.run(init)
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = Y_train_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

                s.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                if j % print_period == 0:
                    test_cost = s.run(
                        cost, feed_dict={
                            X: X_test, T: Y_test_ind})
                    prediction = s.run(predict_op, feed_dict={X: X_test})
                    err = error_rate(prediction, Y_test)
                    print(
                        f"cost / err at iter i={i}, j= {j}: {test_cost}, {err}")
                    ll.append(test_cost)
    plt.plot(ll)
    plt.show()


if __name__ == '__main__':
    main()
