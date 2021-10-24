import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC

dataset = pd.read_csv("C:\\Users\\spmn9\\PycharmProjects\\II_Lab\\spam.csv", encoding='latin_1')

plt.figure(figsize=(6,6))
target = pd.value_counts(dataset["v1"])
target.plot(kind = "pie")
plt.title("Pie chart")
plt.ylabel("")

ham_words = Counter("".join(dataset[dataset["v1"]=="ham"]["v2"]). \
split()).most_common(20)
df_ham_words = pd.DataFrame.from_dict(ham_words)
df_ham_words = df_ham_words.rename(columns={0: "words in non-spam",1:"count"})
df_ham_words.plot.bar(legend = False)
y_pos = np.arange(len(df_ham_words["words in non-spam"]))
plt.xticks(y_pos, df_ham_words["words in non-spam"])
plt.title("more frequent words in non-spam messages")
plt.xlabel("words")
plt.ylabel("number")

tokenizer = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = tokenizer.fit_transform(dataset["v2"])

dataset["v1"] = dataset["v1"].map({"spam":1, "ham":0})
X_train, X_test, y_train, y_test = model_selection \
.train_test_split(X, dataset['v1'], test_size = 0.33)

alpha_range = np.arange(0.1, 20, 0.1)
accuracy_array = []
accuracy_train_array = []
recall_array = []
precision_array = []


for i in alpha_range:
    model = MultinomialNB(alpha=i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)

    accuracy = metrics.accuracy_score(y_test, prediction)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)

    accuracy_array.append(accuracy)
    accuracy_train_array.append(accuracy_train)
    recall_array.append(recall)
    precision_array.append(precision)

    # print("{:.1f}".format(i)
    #       + ": accuracy = " + "{:.5f}".format(accuracy)
    #       + ", recall = " + "{:.3f}".format(recall)
    #       + ", precision = " + "{:.5f}".format(precision))

#### Определение лучшего показателя alpha
matrix = np.matrix(np.c_[alpha_range, accuracy_array, recall_array, precision_array])
models = pd.DataFrame(data = matrix, columns = ['alpha', 'test accuracy', 'test recall', 'test precision'])
best_index = models['test precision'].idxmax()
best_index = models[models['test precision']==precision_array[best_index]]['test accuracy'].idxmax()
print(matrix)
print("\nЛучший показатель alpha = " + "{:.1f}".format(alpha_range[best_index]))
print("{:.1f}".format(alpha_range[best_index])
      + ": accuracy = " + "{:.5f}".format(accuracy_array[best_index])
      + ", recall = " + "{:.3f}".format(recall_array[best_index])
      + ", precision = " + "{:.5f}".format(precision_array[best_index]))

#### Построение зависимости accuracy от alpha
plt.figure(figsize=(6,6))
plt.plot(alpha_range, accuracy_array, label = "test")
plt.plot(alpha_range, accuracy_train_array, label = "train")
plt.title("Accuracy")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.legend()
plt.grid()

#### Построение матрицы ошибок
model = MultinomialNB(alpha = alpha_range[best_index])
model.fit(X_train, y_train)

confusion_matrix_B = confusion_matrix(y_test, model.predict(X_test))
cm_data_frame = pd.DataFrame(data = confusion_matrix_B,
                             columns = ['predicted ham','predicted spam'],
                             index = ['actual ham', 'actual spam'])
print("\nМатрица ошибок:")
print(cm_data_frame)

#### Расчёт ROC кривой
y_pred_pr = model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics. roc_curve (y_test, y_pred_pr)

#### Расчёт метрики AUC-ROC
roc_auc = metrics.auc(fpr, tpr)
print("\nМетрика AUC_ROC = " + "{:.3f}".format(roc_auc))

#### Построение ROC кривой
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()

#### Метод опорных векторов
print("\nМетод опорных векторов")
C_range = np.arange(0.1, 3, 0.1)
accuracy_array_SVC = []
accuracy_train_array_SVC = []
recall_array_SVC = []
precision_array_SVC = []


for i in C_range:
    model = SVC(C=i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)

    accuracy = metrics.accuracy_score(y_test, prediction)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)

    accuracy_array_SVC.append(accuracy)
    accuracy_train_array_SVC.append(accuracy_train)
    recall_array_SVC.append(recall)
    precision_array_SVC.append(precision)

#### Определение лучшего показателя С
matrix_SVC = np.matrix(np.c_[C_range, accuracy_array_SVC, recall_array_SVC, precision_array_SVC])
models = pd.DataFrame(data = matrix_SVC, columns = ['С', 'test accuracy', 'test recall', 'test precision'])
best_index_SVC = models['test precision'].idxmax()
best_index_SVC = models[models['test precision'] == precision_array_SVC[best_index_SVC]]['test accuracy'].idxmax()
print(matrix_SVC)
print("\nЛучший показатель С = " + "{:.1f}".format(C_range[best_index_SVC]))
print("{:.1f}".format(C_range[best_index_SVC])
      + ": accuracy = " + "{:.5f}".format(accuracy_array_SVC[best_index_SVC])
      + ", recall = " + "{:.3f}".format(recall_array_SVC[best_index_SVC])
      + ", precision = " + "{:.5f}".format(precision_array_SVC[best_index_SVC]))

#### Построение зависимости accuracy от C
plt.figure(figsize=(6,6))
plt.plot(C_range, accuracy_array_SVC, label = "test")
plt.plot(C_range, accuracy_train_array_SVC, label = "train")
plt.title("Accuracy")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.grid()

#### Построение матрицы ошибок
model_SVC = SVC(C = C_range[best_index_SVC],
                probability=True)
model_SVC.fit(X_train, y_train)

confusion_matrix_SVC = confusion_matrix(y_test, model.predict(X_test))
cm_data_frame = pd.DataFrame(data = confusion_matrix_SVC,
                             columns = ['predicted ham','predicted spam'],
                             index = ['actual ham', 'actual spam'])
print("\nМатрица ошибок:")
print(cm_data_frame)

#### Расчёт ROC кривой
y_pred_pr = model_SVC.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics. roc_curve (y_test, y_pred_pr)

#### Расчёт метрики AUC-ROC
roc_auc = metrics.auc(fpr, tpr)
print("\nМетрика AUC_ROC = " + "{:.3f}".format(roc_auc))

#### Построение ROC кривой
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()

plt.show()