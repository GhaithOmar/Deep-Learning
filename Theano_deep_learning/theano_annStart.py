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

    max_itr = 20
    reg = 0.01
    lr = 1e-4
    N, D = Xtrain.shape
    batch_sz = 500
    nbatches = N // batch_sz
    M = 300
    K = len(set(Ytrain))

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    thX = T.matrix('X')
    thT = T.matrix('T')

    # Initialize the weights values

    init_W1 = np.random.randn(D, M) / np.sqrt(M)
    init_b1 = np.zeros(M)
    init_W2 = np.random.randn(M, K) / np.sqrt(M)
    init_b2 = np.zeros(K)

    # Create Weights variables

    W1 = theano.shared(init_W1, 'W1')
    b1 = theano.shared(init_b1, 'b1')
    W2 = theano.shared(init_W2, 'W2')
    b2 = theano.shared(init_b2, 'b2')

    # Create Activation equation, output equation, cost
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2) + b2)
    cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + \
             (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
    prediction = T.argmax(thY, axis=1)
    # Create update equation
    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_W1 = W1 - lr * gW1
    update_b1 = b1 - lr * gb1
    update_W2 = W2 - lr * gW2
    update_b2 = b2 - lr * gb2

    # Train function and prediction function
    train_op = theano.function(
        inputs=[
            thX, thT], updates=[
            (W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)])

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
