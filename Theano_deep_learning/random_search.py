import theano.tensor as T
from theano_ann import ANN
from util import get_spiral, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np


def random_search():

    X, Y = get_clouds()
    X, Y = shuffle(X, Y)

    Ntrain = int(0.7 * len(X))

    Xtrain, Xtest = X[:Ntrain], X[Ntrain:]
    Ytrain, Ytest = Y[:Ntrain], Y[Ntrain:]

    M = 20
    nHidden = 2
    log_lr = -4
    log_l2 = -2
    max_tries = 30
    best_nHidden = None
    best_validation_rate = 0
    best_M = None
    best_lr = None
    best_l2 = None
    for _ in range(max_tries):
        model = ANN([M] * nHidden)
        model.fit(
            Xtrain, Ytrain,
            learning_rate=10**log_lr,
            reg=10**log_l2,
            mu=0.99,
            epochs=3000,
            show_fig=False
        )
        validation_accuracy = model.score(Xtest, Ytest)
        train_accuracy = model.score(Xtrain, Ytrain)
        print(
            f'validation_accuracy: {validation_accuracy}, train_accuracy: {train_accuracy}, setting M: {M}, nHidden: {nHidden}, lr: {log_lr}, l2: {log_l2}')

        if validation_accuracy > best_validation_rate:
            best_validation_rate = validation_accuracy
            best_M = M
            best_nHidden = nHidden
            best_l2 = log_l2
            best_lr = log_lr

        # select new hyperparams
        nHidden = best_nHidden + np.random.randint(-1, 2)
        nHidden = max(1, nHidden)
        M = best_M + np.random.randint(-1, 2) * 10
        M = max(10, M)
        log_lr = best_lr + np.random.randint(-1, 2)
        log_l2 = best_l2 + np.random.randint(-1, 2)

    print(f'Best validation_accuracy: {best_validation_rate}')
    print('Best Setting:')
    print(f'Best hidden layer number: {best_nHidden}')
    print(f"Best hidden_layer_size, {best_M}")
    print(f'Best learning rate: {best_lr}')
    print(f'Best regularizations : {best_l2}')


if __name__ == '__main__':
    random_search()
