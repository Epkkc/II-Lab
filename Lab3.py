import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from sklearn.metrics import f1_score
import jinja2

### 1. Импорт данных
dataset = pd.read_csv("C:\\Users\\spmn9\\PycharmProjects\\II_Lab\\income.csv")

### 2.1. Определение признаков в которых есть пропущенные значения и подсчёт количества нулевых ячеек

# attributes_with_null = dataset.isin([' ?'])
# null_sum = attributes_with_null.sum()
# print("Признаки и количество пропущенных значений")
# print(str(null_sum) + "\n")

### 2.2. Построить гистограмму объектов по признаку «workclass»

# plt.figure(figsize=(12, 6))
# bins = np.arange(10) - 0.5
# plt.hist(dataset['workclass'], bins)
# plt.title("Histogram of objects according to \"workclass\" attribute")

### 2.3. Визуализировать совмещенные гистограммы объектов по признаку
###      «income» для двух значений признака «sex» на одном графике

# f, ax = plt.subplots(figsize=(10, 6))
# ax = sns.countplot(x="income", hue="sex", data=dataset, palette="Set1")
# ax.set_title("Frequency distribution of income variable wrt sex")

### 2.4. Визуализировать совмещенные гистограммы объектов по признаку
###      «income» для всех значений признака «race» на одном графике

# f, ax = plt.subplots(figsize=(10, 6))
# ax = sns.countplot(x="income", hue="race", data=dataset, palette="Set1")
# ax.set_title("Frequency distribution of income variable wrt race")

### 2.5. Визуализировать совмещенные гистограммы объектов по признаку
###      «workclass» для двух значений признака «income» на одном графике

# f, ax = plt.subplots(figsize=(10, 6))
# ax = sns.countplot(x="workclass", hue="income", data=dataset, palette="Set1")
# ax.set_title("Frequency distribution of workclass variable wrt income")

### 2.6. Визуализировать совмещенные гистограммы объектов по признаку
###      «workclass» для двух значений признака «sex» на одном графике

# f, ax = plt.subplots(figsize=(10, 6))
# ax = sns.countplot(x="workclass", hue="sex", data=dataset, palette="Set1")
# ax.set_title("Frequency distribution of workclass variable wrt sex")

### 2.7. Визуализировать гистограмму объектов по признаку «age»

# f, ax = plt.subplots(figsize=(10,8))
# x = dataset['age']
# ax = sns.distplot(x, bins=10, color='blue')
# ax.set_title("Distribution of age variable")

### 2.8. Визуализировать распределение объектов по признаку «age», используя «ящик с усами»

# f, ax = plt.subplots(figsize=(10,8))
# x = dataset['age']
# ax = sns.boxplot(x)
# ax.set_title("Visualize outliers in age variable")

### 2.9. Визуализировать два «ящика с усами» по признаку «age» для
###      двух значений признака «income» на одном графике

# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="income", y="age", data=dataset)
# ax.set_title("Visualize income wrt age variable")

### 2.10. Визуализировать четыре «ящика с усами» по признаку «age» для двух
###       значений признака «income» и двух значений признака «sex» на одном графике;

# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="income", y="age", hue="sex", data=dataset)
# ax.set_title("Visualize income wrt age and sex variable")
# ax.legend(loc='upper right')

### 2.11. Визуализировать «ящики с усами» для каждого из значений
###       признака «race» по признаку «аge»

# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="race", y="age", data=dataset)
# ax.set_title("Visualize age wrt race variable")
# ax.legend(loc='upper right')

### 2.12. Визуализировать тепловую карту корреляции признаков

# plt.figure(figsize=(8, 8))
# sns.heatmap(dataset.corr(), annot = True)

### 2.13. Если в наборе данных пропущенные значения обозначены специальным
###       символом, замените значения в таких ячейках на тип NaN

# dataset.replace(' ?', np.NaN, inplace=True)

### 2.14. Определить категориальные признаки в наборе данных
# categorical = [var for var in dataset.columns if dataset[var].dtype == np.object0]
# categorical_attrs = dataset[categorical].head()
# print("Категориальные признаки\n" + str(categorical) + "\n")

### 2.15. Определить числовые признаки в наборе данных
# numeric = [var for var in dataset.columns if dataset[var].dtype == np.int64]
# numeric_attrs = dataset[numeric].head()
# print("Числовые признаки\n" + str(numeric) + "\n")

### 3. Разделить выборку на тренировочную и тестовую по целевой переменной «income»
# attrs = pd.get_dummies(dataset.drop(columns=['income']))
# target = dataset['income']
# target = np.where(target == ' <=50K', 0, 1)
#
# X_train, X_test, y_train, y_test = model_selection \
#     .train_test_split(attrs, target)

### 4. Обучить модель решающего дерева для задачи классификации,
#      построить графики зависимости F-меры на обучающей выборке и на тестовой
#      от глубины дерева. Найти оптимальную глубину дерева, варьируя ее в
#      выбранном диапазоне. Построить для оптимальной модели матрицу ошибок

# print("Модель решающего дерева\n")
# deep = np.arange(2, 100, 2)
# f_values_dec_tree_train = []
# f_values_dec_tree_test = []
#
# for i in deep:
#     model = DecisionTreeClassifier(max_depth=i)
#     model.fit(X_train, y_train)
#     prediction_train = model.predict(X_train)
#     prediction_test = model.predict(X_test)
#     # Для тестовой выборки
#     recall = metrics.recall_score(y_test, prediction_test)
#     precision = metrics.precision_score(y_test, prediction_test)
#     f_values_dec_tree_test.append(2 * precision * recall / (precision + recall))
#     # Для тренировочной выборки
#     recall = metrics.recall_score(y_train, prediction_train)
#     precision = metrics.precision_score(y_train, prediction_train)
#     f_values_dec_tree_train.append(2 * precision * recall / (precision + recall))
#
# plt.figure(figsize=(8, 8))
# plt.plot(deep, f_values_dec_tree_train, label="F train")
# plt.plot(deep, f_values_dec_tree_test, label="F test")
# plt.title("F values decision tree")
# plt.xlabel("deep")
# plt.ylabel("F value")
# plt.legend()

# Построение матрицы ошибок

# best_index = f_values_dec_tree_test.index(max(f_values_dec_tree_test))
# print("Лучшая глубина решающего дерева = " + str(deep[best_index]))
#
# best_model = DecisionTreeClassifier(max_depth=deep[best_index])
# best_model.fit(X_train, y_train)
# confusion_matrix_dec_tree = confusion_matrix(y_test, best_model.predict(X_test))
# cm_data_frame_dec_tree = pd.DataFrame(data=confusion_matrix_dec_tree,
#                                       columns=['actual <=50K', 'actual >50K'],
#                                       index=['predicted <=50K', 'predicted >50K'])
#
# print("Матрица ошибок для модели решающего дерева")
# print(str(cm_data_frame_dec_tree) + "\n")

### 5. Обучить модель случайного леса для задачи классификации,
# построить графики зависимости F-меры на обучающей выборке и на тестовой
# от количества деревьев в композиции. Найдите оптимальное количество
# деревьев в композиции, варьируя их в выбранном диапазоне. Построить для
# оптимальной модели матрицу ошибок.

# print("Модель случайного леса\n")
# trees_number = np.arange(1, 200, 5)
# f_values_ran_for_train = []
# f_values_ran_for_test = []
#
# for i in trees_number:
#     model = RandomForestClassifier(n_estimators=i)
#     model.fit(X_train, y_train)
#     prediction_train = model.predict(X_train)
#     prediction_test = model.predict(X_test)
#     # Для тестовой выборки
#     recall = metrics.recall_score(y_test, prediction_test)
#     precision = metrics.precision_score(y_test, prediction_test)
#     f_values_ran_for_test.append(2 * precision * recall / (precision + recall))
#     # Для тренировочной выборки
#     recall = metrics.recall_score(y_train, prediction_train)
#     precision = metrics.precision_score(y_train, prediction_train)
#     f_values_ran_for_train.append(2 * precision * recall / (precision + recall))
#
# plt.figure(figsize=(8, 8))
# plt.plot(trees_number, f_values_ran_for_train, label="F train")
# plt.plot(trees_number, f_values_ran_for_test, label="F test")
# plt.title("F values random forest")
# plt.xlabel("Number of trees")
# plt.ylabel("F value")
# plt.legend()

# Построение матрицы ошибок

# best_index = f_values_ran_for_test.index(max(f_values_ran_for_test))
# print("Оптимальное количество деревьев = " + str(trees_number[best_index]))
#
# best_model = RandomForestClassifier(n_estimators=trees_number[best_index])
# best_model.fit(X_train, y_train)
# confusion_matrix_ran_for = confusion_matrix(y_test, best_model.predict(X_test))
# cm_data_frame_ran_for = pd.DataFrame(data=confusion_matrix_ran_for,
#                                       columns=['actual <=50K', 'actual >50K'],
#                                       index=['predicted <=50K', 'predicted >50K'])
#
# print("Матрица ошибок для модели случайного леса")
# print(str(cm_data_frame_ran_for))

### 6. Обучить модель градиентного бустинга для задачи классификации,
# построить графики зависимости F-меры на обучающей выборке и на тестовой
# от количества деревьев в композиции. Найдите оптимальное количество
# деревьев в композиции, варьируя их в выбранном диапазоне. Построить для
# оптимальной модели матрицу ошибок.

# print("Модель градиентного бустинга\n")
# trees_number = np.arange(1, 200, 5)
# f_values_ran_for_train = []
# f_values_ran_for_test = []
#
# for i in trees_number:
#     model = CatBoostClassifier(n_estimators=i, verbose=False)
#     model.fit(X_train, y_train)
#     prediction_train = model.predict(X_train)
#     prediction_test = model.predict(X_test)
#     # Для тестовой выборки
#     recall = metrics.recall_score(y_test, prediction_test)
#     precision = metrics.precision_score(y_test, prediction_test)
#     f_values_ran_for_test.append(2 * precision * recall / (precision + recall))
#     # Для тренировочной выборки
#     recall = metrics.recall_score(y_train, prediction_train)
#     precision = metrics.precision_score(y_train, prediction_train)
#     f_values_ran_for_train.append(2 * precision * recall / (precision + recall))
#
# plt.figure(figsize=(8, 8))
# plt.plot(trees_number, f_values_ran_for_train, label="F train")
# plt.plot(trees_number, f_values_ran_for_test, label="F test")
# plt.title("F values gradient boost")
# plt.xlabel("Number of trees")
# plt.ylabel("F value")
# plt.legend()

# Построение матрицы ошибок

# best_index = f_values_ran_for_test.index(max(f_values_ran_for_test))
# best_catboost_trees = trees_number[best_index]
# print("Оптимальное количество деревьев = " + str(trees_number[best_index]))
# best_catboost = CatBoostClassifier(n_estimators=trees_number[best_index], verbose=False)
# best_catboost.fit(X_train, y_train)
# confusion_matrix_ran_for = confusion_matrix(y_test, best_catboost.predict(X_test))
# cm_data_frame_ran_for = pd.DataFrame(data=confusion_matrix_ran_for,
#                                       columns=['actual <=50K', 'actual >50K'],
#                                       index=['predicted <=50K', 'predicted >50K'])
#
# print("Матрица ошибок для модели градиентного бустинга")
# print(str(cm_data_frame_ran_for))


### 7. Обучение модели многослойного перцепторна
# 7.1. Подготовить данные для обучения нейросети

# Замена всех NaN на самые часто встречающиеся значения в колонке

# filled_attrs = dataset.drop(columns=['income'])
# columns = filled_attrs.columns
# for i in columns:
#     filled_attrs[i].fillna(filled_attrs[i].mode()[0], inplace=True)

# Преобразование всех категориальных признаков в числовые\

# filled_attrs_dum = pd.get_dummies(filled_attrs)

# Масштабирование

# scaler = StandardScaler()
# attrs_scaled = scaler.fit_transform(filled_attrs_dum)
#
# X_train, X_test, y_train, y_test = model_selection \
#     .train_test_split(attrs_scaled, target)
#
# y_train = np_utils.to_categorical(y_train, 2)
# y_test = np_utils.to_categorical(y_test, 2)

# 7.2. Обучить модель многослойного перцептрона для задачи
# классификации с оптимальными параметрами, построить графики
# зависимости F-меры на обучающей выборке и на тестовой от количества
# эпох обучения. Определите оптимальное количество эпох обучения и
# архитектуру нейросети. Построить для оптимальной модели матрицу ошибок

# NB_CLASSES = y_train.shape[1]
# INPUT_SHAPE = (X_train.shape[1],)
# perceptron = Sequential()
# perceptron.add(Dense(32, input_shape=INPUT_SHAPE))
# perceptron.add(Activation('relu'))
# perceptron.add(Dropout(0.3))
# perceptron.add(Dense(16))
# perceptron.add(Activation('relu'))
# perceptron.add(Dense(8))
# perceptron.add(Activation('relu'))
# perceptron.add(Dense(NB_CLASSES))
# perceptron.add(Activation('softmax'))
# perceptron.summary()
#
# perceptron.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['Precision', 'Recall'])
#
# EPOCHS = 30
# epochs = np.arange(1, EPOCHS+1, 1)
# history = perceptron.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
#
# f1_score_list_train = []
# f1_score_list_test = []
# for i in range(EPOCHS):
#     f1_score_list_train.append(2 * history.history['precision'][i] * history.history['recall'][i] / (
#             history.history['precision'][i] +
#             history.history['recall'][i]))
#     f1_score_list_test.append(2 * history.history['val_precision'][i] * history.history['val_recall'][i] / (
#             history.history['val_precision'][i] +
#             history.history['val_recall'][i]))
#
#
# # Построение графика Fмера(EPOCHS)
# plt.figure(figsize=(8, 8))
# plt.plot(epochs, f1_score_list_train, label="F train")
# plt.plot(epochs, f1_score_list_test, label="F test")
# plt.title("F values perceptron")
# plt.xlabel("Number of epochs")
# plt.ylabel("F value")
# plt.legend()

# Построение матрицы ошибок

# predict_x = perceptron.predict(X_test)
# classes_x=np.argmax(predict_x,axis=1)
# y_test_test = y_test[:, 1].T
# confusion_matrix_preceptron = confusion_matrix(y_test_test, classes_x)
# cm_data_frame_perceptron = pd.DataFrame(data=confusion_matrix_preceptron,
#                                       columns=['actual <=50K', 'actual >50K'],
#                                       index=['predicted <=50K', 'predicted >50K'])
# print("Матрица ошибок для перцептрона при EPOCHS = " + str(EPOCHS))
# print(str(cm_data_frame_perceptron))


### 8. Повторите пункты 6-7 для набора данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
samples = np.arange(1, 5, 1)
fig = plt.figure(figsize=(8, 8))
for i in samples:
    image = X_train[i]
    if i % 2 == 1:
        plt.subplot2grid((2, 2), ((i-1)//2, 0))
    else:
        plt.subplot2grid((2, 2), ((i-2)//2, 1))
    plt.title("Sample " + str(i))
    plt.imshow(image, cmap='gray')

# Приводим думерный тензор к одномерному вектору
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

# Обучаем градиентный бустинг

print("Модель градиентного бустинга\n")
trees_number = np.arange(1, 100, 4)
f1_scores_train = []
f1_scores_test = []

for i in trees_number:
    print("/Test/Кол-во деревьев = " + str(i))
    model = CatBoostClassifier(n_estimators=i, verbose=False)
    model.fit(X_train, y_train)
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
    # Для тестовой выборки
    f1_scores_train.append(f1_score(y_train, model.predict(X_train), average='micro'))
    # Для тренировочной выборки
    f1_scores_test.append(f1_score(y_test, model.predict(X_test), average='micro'))


plt.figure(figsize=(8, 8))
plt.plot(trees_number, f1_scores_train, label="F train")
plt.plot(trees_number, f1_scores_test, label="F test")
plt.title("F values gradient boost MNIST")
plt.xlabel("Number of trees")
plt.ylabel("F value")
plt.legend()

# Построение матрицы ошибок
best_index = f1_scores_test.index(max(f1_scores_test))
print("Оптимальное количество деревьев = " + str(trees_number[best_index]))
best_catboost = CatBoostClassifier(n_estimators=trees_number[best_index], verbose=False)
best_catboost.fit(X_train, y_train)
confusion_matrix_catboost = confusion_matrix(y_test, best_catboost.predict(X_test))
cm = pd.DataFrame(data = confusion_matrix_catboost,
                  columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                  index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.figure(figsize=(8, 8))
plt.title("Матрица ошибок для модели градиентного бустинга MNIST")
ax = sns.heatmap(cm, annot=True, fmt="d")
print("Матрица ошибок для модели градиентного бустинга MNIST")
print(str(cm))

# Обучаем нейросеть

# Преобразуем в категориальные признаки
y_train_categ = np_utils.to_categorical(y_train, 10)
y_test_categ = np_utils.to_categorical(y_test, 10)

NB_CLASSES = y_train_categ.shape[1]
INPUT_SHAPE = (X_train.shape[1],)
perceptron = Sequential()
perceptron.add(Dense(32, input_shape=INPUT_SHAPE))
perceptron.add(Activation('relu'))
perceptron.add(Dropout(0.3))
perceptron.add(Dense(16))
perceptron.add(Activation('relu'))
perceptron.add(Dense(8))
perceptron.add(Activation('relu'))
perceptron.add(Dense(NB_CLASSES))
perceptron.add(Activation('softmax'))
perceptron.summary()

perceptron.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['Precision', 'Recall'])

EPOCHS = 30
epochs = np.arange(1, EPOCHS + 1, 1)
history = perceptron.fit(X_train, y_train_categ, batch_size=32, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test_categ))

f1_score_list_train = []
f1_score_list_test = []
for i in range(EPOCHS):
    f1_score_list_train.append(2 * history.history['precision'][i] * history.history['recall'][i] / (
            history.history['precision'][i] +
            history.history['recall'][i]))
    f1_score_list_test.append(2 * history.history['val_precision'][i] * history.history['val_recall'][i] / (
            history.history['val_precision'][i] +
            history.history['val_recall'][i]))

# Построение графика Fмера(EPOCHS)
plt.figure(figsize=(8, 8))
plt.plot(epochs, f1_score_list_train, label="F train")
plt.plot(epochs, f1_score_list_test, label="F test")
plt.title("F values perceptron")
plt.xlabel("Number of epochs")
plt.ylabel("F value")
plt.legend()

# Построение матрицы ошибок

predict_x = perceptron.predict(X_test)
classes_x = np.argmax(predict_x, axis=1)
confusion_matrix_preceptron = confusion_matrix(y_test, classes_x)
cm = pd.DataFrame(data = confusion_matrix_preceptron,
                  columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                  index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.figure(figsize=(8, 8))
plt.title("Матрица ошибок для перцептрона MNIST")
ax = sns.heatmap(cm, annot=True, fmt="d")
print("Матрица ошибок для перцептрона MNIST")
print(str(cm))

plt.show()
