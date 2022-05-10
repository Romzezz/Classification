#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from time import time
import numpy as np


# In[2]:


def logistic_regression(X_train, y_train, X_test, regularization=None, visualize=False):
    '''
    Классификатор, основанный на логистической регрессии.
            Параметры:
                    X_train (np.ndarray): 
                        массив признаков обучающей выборки
                    y_train (np.array): 
                        Вектор меток целевого признака обучающей выборки
                    X_test (np.ndarray): 
                        массив признаков тестовой выборки
                    y_test (np.array): 
                        Вектор меток целевого признака тестовой выборки
                    regularization {"l1", "l2", "None"}, default='None':
                        Параметр регуляризации
                    visualize (bool), default=False:
                        Если True, строит график классификации
                        
            Возвращаемое значение:
                    answer (dict):
                        Словарь, в котором хранятся предсказанные метки для тестовой выборки,
                        а также массив весов признаков
    '''
    if regularization is None:
        regularization = 'none'
    answer = dict()
    clf = LogisticRegression(solver='liblinear', penalty=regularization).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    answer['predicted'] = y_pred
    coefs = clf.coef_[0]
    bias = clf.intercept_[0]
    answer['coefs'] = coefs
    answer['bias'] = bias
    if visualize and X_train.shape[1] == 2:
        plt.figure()
        plt.title('visualization of logistic regression')
        xmin, xmax = X_test[:, 0].min() - 1, X_test[:, 1].max() + 1
        ymin, ymax = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        ax = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = -(x_vals * coefs[0] + bias)/coefs[1]
        plt.plot(x_vals, y_vals, '--', c="red")
    return answer


# In[3]:


def svm(X_train, y_train, X_test, visualize=False):
    '''
    Классификатор, основанный на векторе опорных векторов.
            Параметры:
                    X_train (np.ndarray): 
                        массив признаков обучающей выборки
                    y_train (np.array): 
                        Вектор меток целевого признака обучающей выборки
                    X_test (np.ndarray): 
                        массив признаков тестовой выборки
                    y_test (np.array): 
                        Вектор меток целевого признака тестовой выборки
                    visualize (bool), default=False:
                        Если True, строит график классификации
                        
            Возвращаемое значение:
                    answer (dict):
                        Словарь, в котором хранятся предсказанные метки для тестовой выборки,
                        а также массив весов признаков
    '''
    answer = dict()
    clf = SVC(kernel='linear').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    answer['predicted'] = y_pred
    coefs = clf.coef_[0]
    bias = clf.intercept_[0]
    answer['coefs'] = coefs
    answer['bias'] = bias
    if visualize and X_train.shape[1] == 2:
        plt.figure()
        plt.title('visualization of logistic regression')
        ax = plt.gca()
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
                   linewidth=1, facecolors='none', edgecolors='k')
    return answer


# In[4]:


def compare(X_train, y_train, X_test, y_test, regularization=None):
    '''
    Сравнивает показатели времени и точности различных классификаторов на одном наборе данных.
            Параметры:
                    X_train (np.ndarray): 
                        массив признаков обучающей выборки
                    y_train (np.array): 
                        Вектор меток целевого признака обучающей выборки
                    X_test (np.ndarray): 
                        массив признаков тестовой выборки
                    y_test (np.array): 
                        Вектор меток целевого признака тестовой выборки
                        
            Возвращаемое значение:
                    None
    '''
    start = time()
    log_reg_predict = logistic_regression(X_train, y_train, X_test, regularization=regularization)['predicted']
    log_reg_duration = time() - start
    start = time()
    svm_predict = svm(X_train, y_train, X_test)['predicted']
    svm_duration = time() - start
    log_reg_accuracy = accuracy_score(y_test, log_reg_predict)
    svm_accuracy = accuracy_score(y_test, svm_predict)
    values = [['accuracy', 'time(ms)'], [log_reg_accuracy, log_reg_duration], [svm_accuracy, svm_duration]]
    fig = go.Figure(data=[go.Table(header=dict(values=['', 'Logistic Regression', 'SVM']),
                                   cells=dict(values=values))])
    fig.show()


# In[13]:


# чтобы получать новые датасеты, можно перезапускать эту ячейку
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[14]:


logistic_regression(X_train, y_train, X_test, regularization='l1', visualize=True)


# In[15]:


svm(X_train, y_train, X_test, visualize=True)


# In[16]:


compare(X_train, y_train, X_test, y_test, regularization='l1')


# In[ ]:




