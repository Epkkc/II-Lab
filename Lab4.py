import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv1D
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import GRU


# lookback - количество выборок в прошлом, которое определяет окно временного ряда,
#            являющееся признаковым описанием объекта
# delay - количество выборок в будущем, которое определяет окно временного ряда, являющееся целевым значением
# min_index, max_index - два индекса массива входных данных, ограничивающих область извлечения данных
# shuffle - параметр, определяющий, будет ли производиться формирование окон в случайном порядке или последовательно
# batch_size - количество окон в пакете данных
# step - шаг формирования окон из исходного временного ряда
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=24, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data.iloc[indices]
            targets[j] = data.iloc[rows[j] + delay]  # [1]
        yield samples, targets


def graphs(history, model, test_generator, model_name):
    epochs = history.epoch
    plt.figure()
    plt.plot(epochs, history.history["mae"], label="mae")
    plt.plot(epochs, history.history["val_mae"], label="val_mae")
    plt.legend()
    plt.title(model_name)

    print("Evaluate " + str(model_name))
    model.evaluate(test_generator, steps=150)

    print("Test " + str(model_name))
    targets = []
    predictions = []
    ME = 0
    MAE = 0
    count = 0
    for sample, target in test_generator:
        for i in range(len(sample)):
            prediction = model.predict(sample)
            ME += target[i] - prediction[i]
            MAE += abs(target[i] - prediction[i])
            targets.append(target[i])
            predictions.append(prediction[i])
            count += 1
            if count == 2000:
                test_gen.close()

    print("ME = " + str(ME / count) + ", MAE = " + str(MAE / count))
    plt.figure()
    plt.plot(targets, label="target")
    plt.plot(predictions, label="predicted")
    plt.legend()
    plt.title(model_name)


# 1. Загрузить набор данных временного ряда

dataset = pd.read_csv("C:\\Users\\spmn9\\PycharmProjects\\II_Lab\\AEP_hourly.csv")

# 2. Нормализовать временной ряд (Приведение к [0;1])

maxV = dataset['AEP_MW'].max()
minV = dataset['AEP_MW'].min()
dataset_norm = dataset
dataset_norm['AEP_MW'] = dataset['AEP_MW'].apply(lambda x: (x - minV) / (maxV - minV))

datasetDF = dataset_norm
datasetDF = pd.DataFrame(datasetDF)
datasetDF = datasetDF.drop(columns=['Datetime'])

# 3. Написать функцию-генератор, формирующую окна из временного ряда (data) с заданными параметрами
# - количество выборок в прошлом, которое определяет окно временного ряда,
#       являющееся признаковым описанием объекта (lookback);
# - количество выборок в будущем, которое определяет окно временного ряда,
#       являющееся целевым значением (delay);
# - два индекса массива входных данных, ограничивающих область
#       извлечения данных (min_index, max_index);
# - параметр, определяющий, будет ли производиться формирование
#       окон в случайном порядке или последовательно (shuffle);
# - количество окон в пакете данных (batch_size);
# - шаг формирования окон из исходного временного ряда (step).


# 4. Инициализировать генераторы для: обучения, валидации,
# тестирования. Данные для валидации использовать для оценки качества
# модели в процессе обучения. Данные для тестирования использовать для
# оценки качества модели после обучения

train_gen = generator(datasetDF, 24, 0, 0, len(datasetDF) // 4 * 2)
val_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 2 + 1, len(datasetDF) // 4 * 3)
test_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 3 + 1, None)


# 5. Проверить качество базового решения задачи прогнозирования
# без привлечения машинного обучения, предполагая, что следующее значение
# временного ряда равно фактическому значению предыдущего шага.
# Рассчитать среднюю и среднюю абсолютную ошибки. Визуализировать
# спрогнозированный и фактические временные ряды на одном графике

MAE = 0
ME = 0
pred = datasetDF['AEP_MW'][len(datasetDF) // 4 * 3 + 1:len(datasetDF['AEP_MW'])-24:1].values
fact = datasetDF['AEP_MW'][len(datasetDF) // 4 * 3 + 25:len(datasetDF['AEP_MW']):1].values
for i in range(len(fact)):
    dif = fact[i] - pred[i]
    ME += dif
    MAE += abs(dif)

print("ME = " + str(ME/len(fact)) + ", MAE = " + str(MAE/len(fact)))

plt.figure(figsize=(8, 6))
plt.plot(fact, label="fact")
plt.plot(pred, label="predicted")
plt.title("Без МО")
plt.legend()
plt.grid()
plt.show()
# 6. Обучить многослойный перцептрон для прогнозирования
# временного ряда. Рассчитать среднюю и среднюю абсолютную ошибки.
# Визуализировать спрогнозированный и фактические временные ряды на
# одном графике

NB_CLASSES = 1
perceptron_6 = Sequential()
perceptron_6.add(Flatten(input_shape=(24, 1)))
perceptron_6.add(Activation('relu'))
perceptron_6.add(Dropout(0.3))
perceptron_6.add(Dense(16))
perceptron_6.add(Activation('relu'))
perceptron_6.add(Dense(8))
perceptron_6.add(Activation('relu'))
perceptron_6.add(Dense(NB_CLASSES))
perceptron_6.add(Activation('sigmoid'))
perceptron_6.summary()

perceptron_6.compile(loss='mse',
                     optimizer='rmsprop',
                     metrics=['mae'])

EPOCHS = 30  # 30
history = perceptron_6.fit(train_gen, steps_per_epoch=70, epochs=EPOCHS, verbose=1, validation_data=val_gen,
                           validation_steps=60)

graphs(history, perceptron_6, test_gen, "Многослойный перцептон")
plt.show()
# 7. Обучить сверточную сеть прогнозирования временного ряда.
# Рассчитать среднюю и среднюю абсолютную ошибки. Визуализировать
# спрогнозированный и фактические временные ряды на одном графике

train_gen = generator(datasetDF, 24, 0, 0, len(datasetDF) // 4 * 2)
val_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 2 + 1, len(datasetDF) // 4 * 3)
test_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 3 + 1, None)

INPUT_SHAPE = (24, 1)
perceptron_7 = Sequential()
perceptron_7.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=INPUT_SHAPE))
perceptron_7.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
perceptron_7.add(Flatten())
perceptron_7.add(Dense(32, activation='relu'))
perceptron_7.add(Dense(15, activation='relu'))
perceptron_7.add(Dense(1))
perceptron_7.summary()

perceptron_7.compile(loss='mse',
                     optimizer='rmsprop',
                     metrics=['mae'])

EPOCHS = 30  # 30 60 60
history = perceptron_7.fit(train_gen, steps_per_epoch=60, epochs=EPOCHS, verbose=1, validation_data=val_gen,
                           validation_steps=70)

graphs(history, perceptron_7, test_gen, "Свёрточная сеть")

# 8. Обучить рекуррентную сеть с одним рекуррентным слоем для
# прогнозирования временного ряда. Рассчитать среднюю и среднюю
# абсолютную ошибки. Визуализировать спрогнозированный и фактические
# временные ряды на одном графике

train_gen = generator(datasetDF, 24, 0, 0, len(datasetDF) // 4 * 2)
val_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 2 + 1, len(datasetDF) // 4 * 3)
test_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 3 + 1, None)

recur_1 = Sequential()
recur_1.add(GRU(32, input_shape=INPUT_SHAPE))
recur_1.add(Dense(1))
recur_1.summary()

recur_1.compile(loss='mse',
                optimizer='rmsprop',
                metrics=['mae'])

EPOCHS = 30
history = recur_1.fit(train_gen, steps_per_epoch=60, epochs=EPOCHS, verbose=1, validation_data=val_gen,
                      validation_steps=70)

graphs(history, recur_1, test_gen, "Рекуррентная сеть")

# 9. Повторить пункт 8 с использованием двух рекуррентных слоев

train_gen = generator(datasetDF, 24, 0, 0, len(datasetDF) // 4 * 2)
val_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 2 + 1, len(datasetDF) // 4 * 3)
test_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 3 + 1, None)

recur_2 = Sequential()
recur_2.add(GRU(32, return_sequences=True, input_shape=INPUT_SHAPE))
recur_2.add(GRU(32, activation='relu'))
recur_2.add(Dense(1))
recur_2.summary()

recur_2.compile(loss='mse',
                optimizer='rmsprop',
                metrics=['mae'])

EPOCHS = 30
history = recur_2.fit(train_gen, steps_per_epoch=60, epochs=EPOCHS, verbose=1, validation_data=val_gen,
                      validation_steps=70)

graphs(history, recur_2, test_gen, "Рекуррентная сеть два слоя")

# 10. Повторить пункт 9 с использованием прореживания

train_gen = generator(datasetDF, 24, 0, 0, len(datasetDF) // 4 * 2)
val_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 2 + 1, len(datasetDF) // 4 * 3)
test_gen = generator(datasetDF, 24, 0, len(datasetDF) // 4 * 3 + 1, None)

recur_3 = Sequential()
recur_3.add(GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=INPUT_SHAPE))
recur_3.add(GRU(32, activation='relu', dropout=0.1, recurrent_dropout=0.5))
recur_3.add(Dense(1))
recur_3.summary()
recur_3.compile(loss='mse',
                optimizer='rmsprop',
                metrics=['mae'])

EPOCHS = 30
history = recur_3.fit(train_gen,
                      steps_per_epoch=60,
                      epochs=EPOCHS, verbose=1,
                      validation_data=val_gen,
                      validation_steps=70)

graphs(history, recur_3, test_gen, "Рекуррентная сеть с прореживаниями")

plt.show()
print()
