from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt

# Генерируем синусоиды циклом
# Генерируем импульсы (последние прямоугольники)
# Генерируем идеальные треугольники
# Закидываем импульсы в фильтр, получаем на выходе сигналы
# Моделировать tf наверное надо в дискретном виде, поскольку у нас дискретные данные

C_range = []
R_range = []
for i in range(0,10):
    for j in range(1,10,2):
        C_range.append(j*(10**(-12+i)))

for i in range(0,9):
    for j in range(1,10,1):
        R_range.append(j+i*10)

topFreq = 50  # Гц
botFreq = 48  # Гц
stepFreq = 0.2  # Гц
samplesInPeriod = 80
frequency = np.arange(botFreq, topFreq, stepFreq)
numberOfPeriods = 250
x = np.arange(0, numberOfPeriods, 1.0 / samplesInPeriod)
system_sin = np.sin(2 * np.pi * x)
x_all = np.arange(1, numberOfPeriods*samplesInPeriod-1, 1)
system_zero_points = []
for i in x_all:
    if ((system_sin[i] > -0.0000005 and system_sin[i] < 0.0000005)):
        system_zero_points.append(1)
    else:
        system_zero_points.append(0)

all_sin = []
zero_points = []
for i in frequency:
    current_sin = np.sin(2 * i / 50.0 * np.pi * x)
    all_sin.append(current_sin)
    current_zero = []
    for i in range(1, len(current_sin)-1, 1):
        if ((current_sin[i] > -0.0000005 and current_sin[i] < 0.0000005)
                or (current_sin[i] < 0.0 and current_sin[i + 1] > 0.0)
                or (current_sin[i] > 0.0 and current_sin[i + 1] < 0.0)):
            current_zero.append(1)
        else:
            current_zero.append(0)
    zero_points.append(current_zero)

## Построение Диаграммы синусов
plt.figure()
for i in range(len(frequency)):
    plt.subplot(len(frequency), 1, i + 1)
    plt.plot(x, system_sin, label="system")
    plt.plot(x, all_sin[i], label=frequency[i])
    plt.xlabel("Number of periods")
    plt.ylabel(round(frequency[i], 1))

plt.figure()
for i in range(len(frequency)):
    plt.subplot(len(frequency), 1, i + 1)
    plt.plot(x_all, system_zero_points, label="system")
    plt.plot(x_all, zero_points[i], label=frequency[i])
    plt.xlabel("Number of periods")
    plt.ylabel(round(frequency[i], 1))


impulses = []
flag = False
for i in range(len(frequency)):
    current_impulse = []
    for j in x_all:
        if (system_zero_points[j-1] == 1):
            flag = not flag
        if (zero_points[i][j-1] == 1):
            flag = not flag
        if flag:
            current_impulse.append(1)
        else:
            current_impulse.append(0)
    impulses.append(current_impulse)

# for i in range(len(frequency)):
#     current_impulse = []
#     for j in x_all:
#         if (system_zero_points[j-1] == 1 or zero_points[i][j-1] == 1):
#             flag = not flag
#         if flag:
#             current_impulse.append(1)
#         else:
#             current_impulse.append(0)
#     impulses.append(current_impulse)

plt.figure()
for i in range(len(frequency)):
    plt.subplot(len(frequency), 1, i + 1)
    plt.plot(x_all, impulses[i], label=frequency[i])
    plt.xlabel("Number of periods")
    plt.ylabel(round(frequency[i], 1))

plt.figure()
plt.subplot(3, 1, 2)
plt.plot(x_all, system_zero_points, color = "blue")
plt.plot(x_all, zero_points[0], color = "red")
plt.subplot(3, 1, 2)
plt.plot(x_all, impulses[0], )
plt.xlabel("Number of periods")


# Создание эталонных пил
deltas = []
for i in frequency:
    current_delta = []
    current_skolj = 50 - i
    currentSamplesPeriod = round(samplesInPeriod * 50/current_skolj)
    for i in x_all:
        currentPoint = i % currentSamplesPeriod
        if currentPoint < currentSamplesPeriod/2:
            current_delta.append(2/currentSamplesPeriod*currentPoint)
        else:
            current_delta.append(-2/currentSamplesPeriod*(currentPoint-currentSamplesPeriod/2)+1)
    deltas.append(current_delta)

plt.figure()
for i in range(len(frequency)):
    plt.subplot(len(frequency), 1, i + 1)
    plt.plot(x_all, deltas[i], label=frequency[i])
    plt.xlabel("Number of periods")
    plt.ylabel(round(frequency[i], 1))


# impulses
# deltas

R = 500 # Ом
C = 9*10**(-6) # Ф
RC = R*C

func = tf([1],[RC**2, 2*RC, 1])
total_func = func * func
# print(total_func)
ts = 1 / (50 * samplesInPeriod)
discrete_func = c2d(total_func, ts)
ts_all = [(i-1) * ts for i in x_all]
best_ans = lsim(discrete_func, impulses[0])
# print(discrete_func)


# ts = 1 / (50 * samplesInPeriod)
# ts_all = [(i-1) * ts for i in x_all]
# best_sum = -1
# best_func = -1
# best_R = -1
# best_C = -1
# best_ans = -1
# for R in R_range:
#     for C in C_range:
#         RC = R * C
#         func = tf([1],[RC**2, 2*RC, 1])
#         total_func = func * func
#         discrete_func = c2d(total_func, ts)
#         answer = lsim(discrete_func, impulses[0])
#
#         current_sum = 0
#         for i in range(0,len(answer[0])):
#             current_sum += (answer[0][i] - deltas[0][i]) ** 2
#
#         if best_sum == -1 or current_sum < best_sum:
#             best_func = total_func
#             best_R = R
#             best_C = C
#             best_ans = answer
#             print("Best Func\n")
#             print(best_func)
#             print("Best R = " + str(best_R) + ", C = " + str(best_C))
#             print("Current sum = " + str(current_sum))
#     print("R = " + str(R))
# print("Best Func\n")
# print(best_func)
# print("Best R = " + str(best_R) + ", C = " + str(best_C))
plt.figure()
plt.plot(ts_all, deltas[0])
plt.plot(best_ans[1], best_ans[0])
plt.xlabel("Number of periods")


plt.show()
# print("Test")
