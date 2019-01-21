import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
Y = df.iloc[:, -1].values

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# model = Sequential()

# model.add(Dense(units=12, kernel_initializer ='uniform', activation= 'relu', input_dim=X_train.shape[1]))
# model.add(Dropout(p=0.2))
# model.add(Dense(units=6, kernel_initializer ='uniform', activation='relu'))
# model.add(Dropout(p=0.3))
# model.add(Dense(units=1, kernel_initializer ='uniform', activation='sigmoid'))
# model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, batch_size=20, epochs=70)

# y_pred = model.predict(X_test)
# print(y_pred)

customer = np.array(
    [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]).astype(np.float64)
customer = sc.transform(customer)
#new_pred = model.predict(customer)


# # Evaluating the ANN using cross_val
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

# def build_classifier():
# 	model = Sequential()
# 	model.add(Dense(units=12, kernel_initializer ='uniform', activation= 'relu', input_dim=X_train.shape[1]))
# 	model.add(Dense(units=6, kernel_initializer ='uniform', activation='relu'))
# 	model.add(Dense(units=1, kernel_initializer ='uniform', activation='sigmoid'))
# 	model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# 	return model

# model = KerasClassifier(build_fn = build_classifier, batch_size=30, epochs=50)
# accuracies = cross_val_score(estimator=model, X= X_train, y=Y_train, cv=10)#cv is number of folds
# print(accuracies)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    model = Sequential()
    model.add(
        Dense(
            units=12,
            kernel_initializer='uniform',
            activation='relu',
            input_dim=X_train.shape[1]))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(
        Dense(
            units=1,
            kernel_initializer='uniform',
            activation='sigmoid'))
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


model = KerasClassifier(build_fn=build_classifier)
parameters = {
    'batch_size': [25, 36],
    'epochs': [100, 150],
    'optimizer': ['adam', 'rmsprop']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring='accuracy',
    cv=10
)

gs = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)
