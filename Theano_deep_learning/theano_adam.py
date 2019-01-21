import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from util import get_normalized_data, y2indicator


def relu(Z):
    return Z * (Z > 0)


def score(pY, T):
    return np.mean(pY == T)


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    Ytrain = Ytrain.astype(np.int32)
    Ytest = Ytest.astype(np.int32)

    max_itr = 20
    reg = 0.01
    lr = 0.001
    N, D = Xtrain.shape
    batch_sz = 500
    nbatches = N // batch_sz
    M = 300
    K = len(set(Ytrain))

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-10

    thX = T.matrix('X')
    thT = T.matrix('T')

    # Initialize the weights values

    init_W1 = np.random.randn(D, M) / np.sqrt(M)
    init_b1 = np.zeros(M)
    init_W2 = np.random.randn(M, K) / np.sqrt(M)
    init_b2 = np.zeros(K)

    mW1_init = np.zeros((D, M))
    mb1_init = np.zeros(M)
    mW2_init = np.zeros((M, K))
    mb2_init = np.zeros(K)

    vW1_init = np.zeros((D, M))
    vb1_init = np.zeros(M)
    vW2_init = np.zeros((M, K))
    vb2_init = np.zeros(K)

    # Create Weights variables
    W1 = theano.shared(init_W1, 'W1')
    b1 = theano.shared(init_b1, 'b1')
    W2 = theano.shared(init_W2, 'W2')
    b2 = theano.shared(init_b2, 'b2')

    mW1 = theano.shared(mW1_init, 'mW1')
    mb1 = theano.shared(mb1_init, 'mb1')
    mW2 = theano.shared(mW2_init, 'mW2')
    mb2 = theano.shared(mb2_init, 'mb2')

    vW1 = theano.shared(vW1_init, 'vW1')
    vb1 = theano.shared(vb1_init, 'vb1')
    vW2 = theano.shared(vW2_init, 'vW2')
    vb2 = theano.shared(vb2_init, 'vb2')

    t = theano.shared(1, 't')

    # Create Activation equation, output equation, cost
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2) + b2)
    cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + \
             (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
    prediction = T.argmax(thY, axis=1)
    # Create update equation
    corr1 = 1 - beta1 ** t
    corr2 = 1 - beta2 ** t
    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    mW1_update = beta1 * mW1 + (1 - beta1) * gW1
    mb1_update = beta1 * mb1 + (1 - beta1) * gb1
    mW2_update = beta1 * mW2 + (1 - beta1) * gW2
    mb2_update = beta1 * mb2 + (1 - beta1) * gb2

    vW1_update = beta2 * vW1 + (1 - beta2) * gW1 * gW1
    vb1_update = beta2 * vb1 + (1 - beta2) * gb1 * gb1
    vW2_update = beta2 * vW2 + (1 - beta2) * gW2 * gW2
    vb2_update = beta2 * vb2 + (1 - beta2) * gb2 * gb2

    update_W1 = W1 - lr * (mW1 / corr1) / np.sqrt((vW1 / corr2) + epsilon)
    update_b1 = b1 - lr * (mb1 / corr1) / np.sqrt((vb1 / corr2) + epsilon)
    update_W2 = W2 - lr * (mW2 / corr1) / np.sqrt((vW2 / corr2) + epsilon)
    update_b2 = b2 - lr * (mb2 / corr1) / np.sqrt((vb2 / corr2) + epsilon)

    t_update = t + 1

    # Train function and prediction function
    train_op = theano.function(
        inputs=[thX, thT],
        updates=[
            (mW1, mW1_update), (mb1, mb1_update),
            (mW2, mW2_update), (mb2, mb2_update),
            (vW1, vW1_update), (vb1, vb1_update),
            (vW2, vW2_update), (vb2, vb2_update),
            (W1, update_W1), (b1, update_b1),
            (W2, update_W2), (b2, update_b2),
            (t, t_update)
        ]
    )

    get_prediction = theano.function(
        inputs=(thX, thT),
        outputs=[cost, prediction]
    )

    ll = []
    for i in range(max_itr):
        for j in range(nbatches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

            train_op(Xbatch, Ybatch)

            if j % 10 == 0:
                cost_val, prediction_value = get_prediction(Xtest, Ytest_ind)
                ll.append(cost_val)
                acc = score(prediction_value, Ytest)
                print(
                    f'at iter {i} in batch {j}.... cost :{cost_val}  .....acc: {acc}')
    _, p = get_prediction(Xtest, Ytest_ind)
    print(f'final acc= {score(p, Ytest)}')
    plt.plot(ll)
    plt.title('Cost')
    plt.show()


if __name__ == '__main__':
    main()
