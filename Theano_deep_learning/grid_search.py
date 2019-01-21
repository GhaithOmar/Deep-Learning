import theano.tensor as T
from theano_ann import ANN
from util import get_spiral, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np


def grid_search():

    X, Y = get_spiral()
    X, Y = shuffle(X, Y)

    Ntrain = int(0.7 * len(X))

    Xtrain, Xtest = X[:Ntrain], X[Ntrain:]
    Ytrain, Ytest = Y[:Ntrain], Y[Ntrain:]

    hidden_layer_sizes = [
        [300],
        [100, 100],
        [200, 200],
        [50, 50, 50]
    ]

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-5]

    regularizations = [0., 0.1, 1.0]
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for m in hidden_layer_sizes:
        for lr in learning_rates:
            for l2 in regularizations:
                model = ANN(m)
                model.fit(
                    Xtrain,
                    Ytrain,
                    learning_rate=lr,
                    reg=l2,
                    mu=0.99,
                    epochs=3000,
                    show_fig=False)
                validation_accuracy = model.score(Xtest, Ytest)
                train_accuracy = model.score(Xtrain, Ytrain)
                print(
                    f'validation_accuracy: {validation_accuracy}, train_accuracy: {train_accuracy}, setting hls,lr,l2: {m}, {lr}, {l2}')

                if validation_accuracy > best_validation_rate:
                    best_validation_rate = validation_accuracy
                    best_hls = m
                    best_l2 = l2
                    best_lr = lr

    print(f'Best validation_accuracy: {best_validation_rate}')
    print('Best Setting:')
    print(f"Best hidden_layer_size, {best_hls}")
    print(f'Best learning rate: {best_lr}')
    print(f'Best regularizations : {best_l2}')


if __name__ == '__main__':
    grid_search()
