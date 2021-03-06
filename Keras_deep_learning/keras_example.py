from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator

import matplotlib.pyplot as plt

Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

N, D = Xtrain.shape
K = len(set(Ytrain))

Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)

model = Sequential()

model.add(Dense(units=500, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(
    Xtrain,
    Ytrain,
    validation_data=(
        Xtest,
        Ytest),
    epochs=15,
    batch_size=32)
print('return:', r)
