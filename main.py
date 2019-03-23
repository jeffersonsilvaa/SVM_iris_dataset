#teste com Iris dataset e SVM
#https://scikit-learn.org/stable/modules/svm.html
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

import numpy as np
from sklearn.svm import SVC

dataset = np.loadtxt("iris.txt", delimiter=",")
print(dataset.shape)

X = dataset[:,0:3]
y = dataset[:,4]


clf = SVC(gamma='auto')
clf.fit(X, y)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=100, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.predict(X)
acuracia = clf.score(X, y)

print('-----------------------------------------------------------')
print('Acc = ', acuracia)
print('-----------------------------------------------------------')