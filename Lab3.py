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
# import jinja2

### 1. Импорт данных
dataset = pd.read_csv("C:\\Users\\spmn9\\PycharmProjects\\II_Lab\\income.csv")

### 2.1. Определение признаков в которых есть пропущенные значения и подсчёт количества нулевых ячеек
attributes_with_null = dataset.isnull()
null_sum = attributes_with_null.sum()
print("Признаки и количество пропущенных значений")
print(str(null_sum) + "\n")

### 2.2. Построить гистограмму объектов по признаку «workclass»

# plt.figure(figsize=(12, 6))
# dataset['workclass'].hist()
# plt.title("Histogram of objects according to \"workclass\" attribute")
# TODO Криво выводится гистограмма????

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
### TODO "distplot()" является deprecated

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
# dataset.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
### TODO: Не получается построить тепловую карту

### 2.13. Если в наборе данных пропущенные значения обозначены специальным
###       символом, замените значения в таких ячейках на тип NaN
dataset.replace('?', np.NaN, inplace=True)

### 2.14. Определить категориальные признаки в наборе данных
categorical = [var for var in dataset.columns if dataset[var].dtype == np.object0]
categorical_attrs = dataset[categorical].head()
print("Категориальные признаки\n" + str(categorical) + "\n")

### 2.15. Определить числовые признаки в наборе данных
numeric = [var for var in dataset.columns if dataset[var].dtype == np.int64]
numeric_attrs = dataset[numeric].head()
print("Числовые признаки\n" + str(numeric) + "\n")

### 3. Разделить выборку на тренировочную и тестовую по целевой переменной «income»
attrs = pd.get_dummies(dataset.drop(columns=['income']))
target = dataset['income']
target = np.where(target == ' <=50K', 0, 1)

X_train, X_test, y_train, y_test = model_selection \
    .train_test_split(attrs, target)

### 4. Обучить модель решающего дерева для задачи классификации,
#      построить графики зависимости F-меры на обучающей выборке и на тестовой
#      от глубины дерева. Найти оптимальную глубину дерева, варьируя ее в
#      выбранном диапазоне. Построить для оптимальной модели матрицу ошибок
print("Модель решающего дерева\n")
deep = np.arange(2, 100, 2)
f_values_dec_tree_train = []
f_values_dec_tree_test = []

for i in deep:
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train, y_train)
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
    # Для тестовой выборки
    recall = metrics.recall_score(y_test, prediction_test)
    precision = metrics.precision_score(y_test, prediction_test)
    f_values_dec_tree_test.append(2 * precision * recall / (precision + recall))
    # Для тренировочной выборки
    recall = metrics.recall_score(y_train, prediction_train)
    precision = metrics.precision_score(y_train, prediction_train)
    f_values_dec_tree_train.append(2 * precision * recall / (precision + recall))

plt.figure(figsize=(8, 8))
plt.plot(deep, f_values_dec_tree_train, label="F train")
plt.plot(deep, f_values_dec_tree_test, label="F test")
plt.title("F values decision tree")
plt.xlabel("deep")
plt.ylabel("F value")
plt.legend()

# Построение матрицы ошибок
best_index = f_values_dec_tree_test.index(max(f_values_dec_tree_test))
print("Лучшая глубина решающего дерева = " + str(deep[best_index]))

best_model = DecisionTreeClassifier(max_depth=deep[best_index])
best_model.fit(X_train, y_train)
confusion_matrix_dec_tree = confusion_matrix(y_test, best_model.predict(X_test))
cm_data_frame_dec_tree = pd.DataFrame(data=confusion_matrix_dec_tree,
                                      columns=['actual <=50K', 'actual >50K'],
                                      index=['predicted <=50K', 'predicted >50K'])

print("Матрица ошибок для модели решающего дерева")
print(str(cm_data_frame_dec_tree) + "\n")

### 5. Обучить модель случайного леса для задачи классификации,
# построить графики зависимости F-меры на обучающей выборке и на тестовой
# от количества деревьев в композиции. Найдите оптимальное количество
# деревьев в композиции, варьируя их в выбранном диапазоне. Построить для
# оптимальной модели матрицу ошибок.

print("Модель случайного леса\n")
trees_number = np.arange(1, 200, 5)
f_values_ran_for_train = []
f_values_ran_for_test = []

for i in trees_number:
    model = RandomForestClassifier(n_estimators=i)
    model.fit(X_train, y_train)
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
    # Для тестовой выборки
    recall = metrics.recall_score(y_test, prediction_test)
    precision = metrics.precision_score(y_test, prediction_test)
    f_values_ran_for_test.append(2 * precision * recall / (precision + recall))
    # Для тренировочной выборки
    recall = metrics.recall_score(y_train, prediction_train)
    precision = metrics.precision_score(y_train, prediction_train)
    f_values_ran_for_train.append(2 * precision * recall / (precision + recall))

plt.figure(figsize=(8, 8))
plt.plot(trees_number, f_values_ran_for_train, label="F train")
plt.plot(trees_number, f_values_ran_for_test, label="F test")
plt.title("F values random forest")
plt.xlabel("Number of trees")
plt.ylabel("F value")
plt.legend()

# Построение матрицы ошибок
best_index = f_values_ran_for_test.index(max(f_values_ran_for_test))
print("Оптимальное количество деревьев = " + str(trees_number[best_index]))

best_model = RandomForestClassifier(n_estimators=trees_number[best_index])
best_model.fit(X_train, y_train)
confusion_matrix_ran_for = confusion_matrix(y_test, best_model.predict(X_test))
cm_data_frame_ran_for = pd.DataFrame(data=confusion_matrix_ran_for,
                                      columns=['actual <=50K', 'actual >50K'],
                                      index=['predicted <=50K', 'predicted >50K'])

print("Матрица ошибок для модели случайного леса")
print(str(cm_data_frame_ran_for))

### 6. Обучить модель градиентного бустинга для задачи классификации,
# построить графики зависимости F-меры на обучающей выборке и на тестовой
# от количества деревьев в композиции. Найдите оптимальное количество
# деревьев в композиции, варьируя их в выбранном диапазоне. Построить для
# оптимальной модели матрицу ошибок.

print("Модель градиентного бустинга\n")
trees_number = np.arange(1, 200, 5)
f_values_ran_for_train = []
f_values_ran_for_test = []

for i in trees_number:
    model = CatBoostClassifier(n_estimators=i, verbose=False)
    model.fit(X_train, y_train)
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
    # Для тестовой выборки
    recall = metrics.recall_score(y_test, prediction_test)
    precision = metrics.precision_score(y_test, prediction_test)
    f_values_ran_for_test.append(2 * precision * recall / (precision + recall))
    # Для тренировочной выборки
    recall = metrics.recall_score(y_train, prediction_train)
    precision = metrics.precision_score(y_train, prediction_train)
    f_values_ran_for_train.append(2 * precision * recall / (precision + recall))

plt.figure(figsize=(8, 8))
plt.plot(trees_number, f_values_ran_for_train, label="F train")
plt.plot(trees_number, f_values_ran_for_test, label="F test")
plt.title("F values gradient boost")
plt.xlabel("Number of trees")
plt.ylabel("F value")
plt.legend()

# Построение матрицы ошибок
best_index = f_values_ran_for_test.index(max(f_values_ran_for_test))
print("Оптимальное количество деревьев = " + str(trees_number[best_index]))

best_model = CatBoostClassifier(n_estimators=trees_number[best_index], verbose=False)
best_model.fit(X_train, y_train)
confusion_matrix_ran_for = confusion_matrix(y_test, best_model.predict(X_test))
cm_data_frame_ran_for = pd.DataFrame(data=confusion_matrix_ran_for,
                                      columns=['actual <=50K', 'actual >50K'],
                                      index=['predicted <=50K', 'predicted >50K'])

print("Матрица ошибок для модели градиентного бустинга")
print(str(cm_data_frame_ran_for))


### 7. Обучение модели многослойного перцепторна
# 7.1. Подготовить данные для обучения нейросети



# 7.2. Обучить модель многослойного перцептрона для задачи
# классификации с оптимальными параметрами, построить графики
# зависимости F-меры на обучающей выборке и на тестовой от количества
# эпох обучения. Определите оптимальное количество эпох обучения и
# архитектуру нейросети. Построить для оптимальной модели матрицу ошибок



### 8. Повторите пункты 6-7 для набора данных MNIST


plt.show()
