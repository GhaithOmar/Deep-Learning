import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b1, derivative_b2


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

    M = 300
    K = 10

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(K)
    b2 = np.zeros(K)

    # copy weights
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # 1. Constant Learning rate

    losses_batch = []
    errors_batch = []

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Y_train_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            W2 -= lr * (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
            b2 -= lr * (derivative_b2(Ybatch, pYbatch) + reg * b2)
            W1 -= lr * (derivative_w1(Xbatch, Z,
                                      Ybatch, pYbatch, W2) + reg * W1)
            b1 -= lr * (derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1)

            if j % print_period == 0:
                pY, _ = forward(X_test, W1, b1, W2, b2)
                l = cost(pY, Y_test_ind)
                losses_batch.append(l)
                print(f'Cost at iteration i={i}, j={j} : {l}')

                e = error_rate(pY, Y_test)
                errors_batch.append(e)
                print("error_rate", e)
    pY, _ = forward(X_test, W1, b1, W2, b2)

    batch_error = error_rate(pY, Y_test)
    print(f"Final batch error rate: {batch_error}")

    # 2. RMSProp

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    losses_RMSP = []
    errors_RMSP = []

    lr0 = 0.001
    cache_W1 = 1
    cache_b1 = 1
    cache_W2 = 1
    cache_b2 = 1
    decay = 0.999
    epsilon = 1e-10

    for i in range(max_iter):

        for j in range(n_batches):
            Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Y_train_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            cache_W2 = decay * cache_W2 + (1 - decay) * gW2 * gW2
            W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + epsilon)

            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            cache_b2 = decay * cache_b2 + (1 - decay) * gb2 * gb2
            b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + epsilon)

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            cache_W1 = decay * cache_W1 + (1 - decay) * gW1 * gW1
            W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + epsilon)

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
            cache_b1 = decay * cache_b1 + (1 - decay) * gb1 * gb1
            b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + epsilon)

            if j % print_period == 0:
                pY, _ = forward(X_test, W1, b1, W2, b2)
                l = cost(pY, Y_test_ind)
                losses_RMSP.append(l)
                print(f'Cost at iteration i={i}, j={j} : {l}')

                e = error_rate(pY, Y_test)
                errors_RMSP.append(e)
                print("error_rate", e)
    pY, _ = forward(X_test, W1, b1, W2, b2)
    print(f"Final RMSProp error rate: {error_rate(pY, Y_test)}")
    print(f"Final batch error rate: {batch_error}")
    plt.plot(losses_batch, label='batch cost')
    plt.plot(losses_RMSP, label='RMSProp cost')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
