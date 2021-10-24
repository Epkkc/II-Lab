import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import *
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("C:\\Users\\spmn9\\PycharmProjects\\II_Lab\\breast_cancer.csv")

## Заменяем буквы на цифры 0 = M, 1 = B
dataset.at[dataset['diagnosis'] == 'M', 'diagnosis'] = 0
dataset.at[dataset['diagnosis'] == 'B', 'diagnosis'] = 1

samples = dataset.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
target_values = dataset['diagnosis']

samples_np = samples.to_numpy(dtype='float32')
target_values_np = target_values.to_numpy(dtype='float32')

max_neighbours_for_test = 50
neighbours_list = np.arange(0, max_neighbours_for_test + 1, 1)
accuracy_data = [0] * max_neighbours_for_test

## По умолчанию 0,25 преобразуется в тестовую выборку, параметр test_size
samples_training, samples_test, targetValues_training, targetValues_test = \
    train_test_split(samples, target_values)

## Преобразование DataFrame в массивы
samples_training_array = samples_training.to_numpy(dtype='float32')
targetValues_training_array = targetValues_training.to_numpy(dtype='float32')
samples_test_array = samples_test.to_numpy(dtype='float32')
targetValues_test_array = targetValues_test.to_numpy(dtype='float32')

for i in range(max_neighbours_for_test):
    neighboursClassifier = KNeighborsClassifier(n_neighbors=(i + 1))
    neighboursClassifier.fit(samples_training_array, targetValues_training_array)
    result_array = neighboursClassifier.predict(samples_test_array)
    accuracy = accuracy_score(targetValues_test_array, result_array)
    accuracy_data[i] = accuracy


plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
x_axis_data = np.arange(1, max_neighbours_for_test + 1, 1)
plt.plot(x_axis_data, accuracy_data)
plt.title("Test before cross val")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.grid()

kFold = KFold(n_splits=5, shuffle=True)
accuracy_data_cross = [0] * max_neighbours_for_test
for i in range(max_neighbours_for_test):
    best_neighbour = KNeighborsClassifier(n_neighbors=(i+1))
    array = cross_val_score(best_neighbour, samples_np, target_values_np, cv=kFold, scoring='accuracy')
    accuracy_data_cross[i] = np.mean(array)

plt.subplot(1, 2, 2)
plt.plot(x_axis_data, accuracy_data_cross)
plt.title("Test after cross val")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.grid()

## Без масштабирования
c_range = np.arange(0.1, 5.0, 0.1)

accuracy_data_regr = [0] * len(c_range)
for i in range(len(c_range)):
    log_regr = LogisticRegression(C=c_range[i], max_iter=10_000)
    regr_array = cross_val_score(log_regr, samples_np, target_values_np, cv=kFold, scoring='accuracy')
    accuracy_data_regr[i] = np.mean(regr_array)

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(c_range, accuracy_data_regr)
plt.title("Lin Regr without masht")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.grid()


## С масштабирвоанием
scaler = StandardScaler() # Посмотреть как с помощью среднего и дисперсии масштабировать
scaled_samples_np = scaler.fit_transform(samples_np)

accuracy_data_regr_with = [0] * len(c_range)

for i in range(len(c_range)):
    log_regr = LogisticRegression(C=c_range[i], max_iter=10_000)
    regr_array = cross_val_score(log_regr, scaled_samples_np, target_values_np, cv=kFold, scoring='accuracy')
    accuracy_data_regr_with[i] = np.mean(regr_array)

plt.subplot(1, 2, 2)
plt.plot(c_range, accuracy_data_regr_with)
plt.title("Lin Regr with masht")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.grid()
plt.show()