import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import fft
from scipy import signal
# вариант 10 Замерець
n = 500
Fs = 1000
Fmax = 21
F_filter = 28
random = numpy.random.normal(0, 10,  n)
time_line_ox = numpy.arange(n)/Fs
w = Fmax/(Fs/2)
parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')
filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, random)
# график 1
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(time_line_ox, filtered_signal, linewidth = 1)
ax.set_xlabel("Время (секунды) ", fontsize = 14)
ax.set_ylabel("Амплитуда сигнала ", fontsize = 14)
plt.title("Сигнал с максимальной частотой Fmax = 21", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 1' + '.png', dpi = 600)
dpi = 600
# расчет для второго графика
spectrum = scipy.fft.fft(filtered_signal)
spectrum = numpy.abs(scipy.fft.fftshift(spectrum))
length_signal = n
freq_countdown = scipy.fft.fftfreq(length_signal, 1/length_signal)
freq_countdown = scipy.fft.fftshift(freq_countdown)
#второй график
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(freq_countdown, spectrum, linewidth = 1)
ax.set_xlabel("Частота (Гц)", fontsize = 14)
ax.set_ylabel("Амплитуда спектра ", fontsize = 14)
plt.title("Спектр сигнала с максимальной частотой Fmax = 21", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 2' + '.png', dpi=600)
# Практична робота №3
discrete_signals = []
steps = (2, 4, 8, 16)
for Dt in steps:
    discrete_signal = numpy.zeros(n)
    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]
    discrete_signals.append(list(discrete_signal))

fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, discrete_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Время (секунди)', fontsize=14)
fig.supylabel('Амплитуда сигнала', fontsize=14)
fig.suptitle(f'Сигнал с шагом дискретизации Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'график 3' + '.png', dpi=600)

discrete_spectrums = []
for Ds in discrete_signals:
    spectrum = fft.fft(Ds)
    spectrum = numpy.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(list(spectrum))

freq_countdown = fft.fftfreq(n, 1 / n)
freq_countdown = fft.fftshift(freq_countdown)
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(freq_countdown, discrete_spectrums[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Частота (Гц) ', fontsize=14)
fig.supylabel('Амплитуда спектра', fontsize=14)
fig.suptitle(f'Спектры сигналов с шагом дискретизации Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'график 4' + '.png', dpi=600)
w = F_filter / (Fs / 2)
parameters_filter = signal.butter(3, w, 'low', output='sos')
filtered_discretes_signal = []
for discrete_signal in discrete_signals:
    discrete_signal = signal.sosfiltfilt(parameters_filter, discrete_signal)
    filtered_discretes_signal.append(list(discrete_signal))
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, filtered_discretes_signal[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Время (секунды)', fontsize=14)
fig.supylabel('Амплитуда сигнала', fontsize=14)
fig.suptitle(f'Востановленные аналоговые сигналы с шагом дискретизации Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'график 5' + '.png', dpi=600)
dispersions = []
signal_noise = []
for i in range(len(steps)):
    E1 = filtered_discretes_signal[i] - filtered_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filtered_signal) / dispersion)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(steps, dispersions, linewidth=1)
ax.set_xlabel('Шаг дискретизации', fontsize=14)
ax.set_ylabel('Дисперсия', fontsize=14)
plt.title(f'Зависимость дисперсии от шага дискретизации', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 6' + '.png', dpi=600)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(steps, signal_noise, linewidth=1)
ax.set_xlabel('Шаг дискретизации', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title(f'Зависимость соотношения сигнал-шум от шага дискретизации', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 7' + '.png', dpi=600)
#PZ 4
bits_list = []
quantize_signals = []
num = 0
for M in [4, 16, 64, 256]:
    delta = (numpy.max(filtered_signal) - numpy.min(filtered_signal)) / (M - 1)
    quantize_signal = delta * np.round(filtered_signal / delta)
    quantize_signals.append(list(quantize_signal))
    quantize_levels = numpy.arange(numpy.min(quantize_signal), numpy.max(quantize_signal) + 1, delta)
    quantize_bit = numpy.arange(0, M)
    quantize_bit = [format(bits, '0' + str(int(numpy.log(M) / numpy.log(2))) + 'b') for bits in quantize_bit]
    quantize_table = numpy.c_[quantize_levels[:M], quantize_bit[:M]]
    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значение сигнала', 'Кодовая последовательность'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig('./figures/' + 'Таблица квантования для %d уровней ' % M + '.png', dpi=600)
    bits = []
    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]
    bits_list.append(bits)
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(numpy.arange(0, len(bits)), bits, linewidth=0.1)
    ax.set_xlabel('Биты', fontsize=14)
    ax.set_ylabel('Амплитуда сигнала', fontsize=14)
    plt.title(f'Кодовая последовательность при количестве уровней квантования {M}', fontsize=14)
    ax.grid()
    fig.savefig('./figures/' + 'График %d ' % (8 + num) + '.png', dpi=600)
    num += 1
dispersions = []
signal_noise = []
for i in range(4):
    E1 = quantize_signals[i] - filtered_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filtered_signal) / dispersion)
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, quantize_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Время (секунды)', fontsize=14)
fig.supylabel('Амплитуда сигнала', fontsize=14)
fig.suptitle(f'Цифровые сигналы с уровнями квантования (4, 16, 64, 256)', fontsize=14)
fig.savefig('./figures/' + 'график 12' + '.png', dpi=600)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], dispersions, linewidth=1)
ax.set_xlabel('Количество уровней квантования', fontsize=14)
ax.set_ylabel('Дисперсия', fontsize=14)
plt.title(f'Зависимость дисперсии от количества уровней квантования', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 13' + '.png', dpi=600)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], signal_noise, linewidth=1)
ax.set_xlabel('Количество уровней квантования', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title(f'Зависимость соотношения сигнал-шум от количества уровней квантования', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 14' + '.png', dpi=600)
print("sign.noise: ", signal_noise)