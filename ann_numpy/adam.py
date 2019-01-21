import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b1, derivative_b2


def main():
    max_iter = 10
    print_period = 10
    X_train, X_test, Y_train, Y_test = get_normalized_data()
    reg = 0.01

    Y_train_ind = y2indicator(Y_train)
    Y_test_ind = y2indicator(Y_test)

    N, D = X_train.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10

    W1_0 = np.random.randn(D, M) / np.sqrt(D)
    b1_0 = np.zeros(M)
    W2_0 = np.random.randn(M, K) / np.sqrt(K)
    b2_0 = np.zeros(K)

    # .1 Adam

    W1 = W1_0.copy()
    W2 = W2_0.copy()
    b1 = b1_0.copy()
    b2 = b2_0.copy()

    losses_adam = []
    errors_adam = []

    # 1st moment
    mW1 = 0
    mW2 = 0
    mb1 = 0
    mb2 = 0

    # 2nd moment
    vW1 = 0
    vW2 = 0
    vb1 = 0
    vb2 = 0

    # Hyperparams
    eps = 1e-8
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999

    t = 1

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Y_train_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates

            # gradients
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1

            # new m
            mW1 = beta1 * mW1 + (1 - beta1) * gW1
            mb1 = beta1 * mb1 + (1 - beta1) * gb1
            mW2 = beta1 * mW2 + (1 - beta1) * gW2
            mb2 = beta1 * mb2 + (1 - beta1) * gb2

            # new v
            vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
            vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
            # bias correction
            correction1 = 1 - beta1 ** t
            mW1_hat = mW1 / correction1
            mb1_hat = mb1 / correction1
            mW2_hat = mW2 / correction1
            mb2_hat = mb2 / correction1

            #
            correction2 = 1 - beta2 ** t
            vb2_hat = vb2 / correction2
            vb1_hat = vb1 / correction2
            vW2_hat = vW2 / correction2
            vW1_hat = vW1 / correction2

            t += 1
            # weights
            W1 = W1 - lr * mW1_hat / np.sqrt(vW1_hat + eps)
            b1 = b1 - lr * mb1_hat / np.sqrt(vb1_hat + eps)
            W2 = W2 - lr * mW2_hat / np.sqrt(vW2_hat + eps)
            b2 = b2 - lr * mb2_hat / np.sqrt(vb2_hat + eps)

            if j % print_period == 0:
                pY, _ = forward(X_test, W1, b1, W2, b2)
                l = cost(pY, Y_test_ind)
                losses_adam.append(l)
                print(f'Adam Cost at iteration i={i}, j={j} : {l}')

                e = error_rate(pY, Y_test)
                errors_adam.append(e)
                print("error_rate", e)

    pY, _ = forward(X_test, W1, b1, W2, b2)
    adam_error = error_rate(pY, Y_test)

    # 3. RMSProp with momentum
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    losses_rms = []
    errors_rms = []

    # comparable hyper parameters for fair
    lr0 = 0.001
    mu = 0.9
    decay_rate = 0.999
    eps = 1e-8

    # rmsprop cache
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1

    # momentum
    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Y_train_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
            dW2 = mu * dW2 + (1 - mu) * lr0 * gW2 / (np.sqrt(cache_W2) + eps)
            W2 -= dW2

            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
            db2 = mu * db2 + (1 - mu) * lr0 * gb2 / (np.sqrt(cache_b2) + eps)
            b2 -= db2

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
            dW1 = mu * dW1 + (1 - mu) * lr0 * gW1 / (np.sqrt(cache_W1) + eps)
            W1 -= dW1

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
            db1 = mu * db1 + (1 - mu) * lr0 * gb1 / (np.sqrt(cache_b1) + eps)
            b1 -= db1
            if j % print_period == 0:
                pY, _ = forward(X_test, W1, b1, W2, b2)
                l = cost(pY, Y_test_ind)
                losses_rms.append(l)
                print(f'Cost at iteration i={i}, j={j} : {l}')
                err = error_rate(pY, Y_test)
                errors_rms.append(err)
                print("Error rate:", err)

    pY, _ = forward(X_test, W1, b1, W2, b2)

    rms_error = error_rate(pY, Y_test)

    print(f"Final RMSProp error rate: {rms_error}")
    print(f"Final Adam error rate: {adam_error}")
    plt.plot(losses_adam, label='batch cost')
    plt.plot(losses_rms, label='RMSProp cost')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
