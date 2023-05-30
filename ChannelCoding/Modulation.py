from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi
import scipy

seq = np.zeros(1000)
def spectrum_seq(sequence):
    y_spectrum = np.abs(scipy.fft.fftshift(scipy.fft.fft(sequence)))
    x_spectrum = scipy.fft.fftshift(scipy.fft.fftfreq(len(sequence), 1 / len(sequence)))
    return x_spectrum[round(len(x_spectrum)/2):], y_spectrum[round(len(y_spectrum)/2):]


def create_sequence():
    global seq
    seq[0:100] = randint(0, 1)
    seq[100:200] = randint(0, 1)
    seq[200:300] = randint(0, 1)
    seq[300:400] = randint(0, 1)
    seq[400:500] = randint(0, 1)
    seq[500:600] = randint(0, 1)
    seq[600:700] = randint(0, 1)
    seq[700:800] = randint(0, 1)
    seq[800:900] = randint(0, 1)
    seq[900:999] = randint(0, 1)
    return seq


def plot(x, y, axis_x="", axis_y="", title=""):
    # Создаем график
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(axis_x, fontsize=14)
    ax.set_ylabel(axis_y, fontsize=14)
    plt.title(title, fontsize=14)

    # Создаем папку для сохранения графиков, если она не существует
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')

    # Сохраняем график в виде изображения
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.close(fig)


def ask_modulation(frequency, sequence):
    # Выполняем амплитудную модуляцию (ASK)
    sequence_ask = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_ask[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)
    return sequence_ask


def ask_demodulation(frequency, sequence):
    # Выполняем демодуляцию амплитудной модуляции (ASK)
    ask_product = np.zeros(1000)
    ask_demodulated_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    sequence_demodulated = np.zeros(1000)

    # Умножаем сигнал на опорную волну
    for i in range(0, len(sequence)):
        ask_product[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)

    # Выполняем интегрирование и детектирование
    for i in range(0, 10):
        S = 0
        for t in range(0, 100):
            S = S + ask_product[t + 100 * (i - 1)]
            ask_demodulated_signal[t + 100 * (i - 1)] = S

    # Применяем пороговую функцию для получения демодулированного сигнала
    ask_demodulated = 1 / 2 * (np.sign(ask_demodulated_signal - threshold) + 1)

    # Восстанавливаем последовательность символов
    for i in range(0, 10):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]

    return ask_demodulated_signal, sequence_demodulated


def psk_modulation(frequency, sequence):
    # Выполняем фазовую модуляцию (PSK)
    sequence_psk = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_psk[i] = sin(2 * pi * frequency * i / 1000 + sequence[i] * pi + pi)
    return sequence_psk


def psk_demodulation(frequency, sequence):
    # Выполняем демодуляцию фазовой модуляции (PSK)
    psk_product = np.zeros(1000)
    psk_demodulated_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    sequence_demodulated = np.zeros(1000)

    # Умножаем сигнал на опорную волну
    for i in range(0, len(sequence)):
        psk_product[i] = sequence[i] * sin(2 * pi * frequency * i / 1000)

    # Выполняем интегрирование и детектирование
    for i in range(0, 10):
        S = 0
        for t in range(0, 100):
            S = S + psk_product[t + 100 * (i - 1)]
            psk_demodulated_signal[t + 100 * (i - 1)] = S

    # Применяем пороговую функцию для получения демодулированного сигнала
    psk_demodulated = 1 / 2 * (np.sign(psk_demodulated_signal - threshold) + 1)

    # Восстанавливаем последовательность символов
    for i in range(0, 10):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = psk_demodulated[100 * i - 1]

    return psk_demodulated_signal, sequence_demodulated


def fsk_modulation(frequency1, frequency2, sequence):
    # Выполняем частотную модуляцию (FSK)
    sequence_fsk = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_fsk[i] = sequence[i] * sin(2 * pi * frequency1 * i / 1000) + (abs(sequence[i] - 1)) * sin(
            2 * pi * frequency2 * i / 1000)
    return sequence_fsk


def fsk_demodulation(frequency1, frequency2, sequence):
    # Выполняем демодуляцию частотной модуляции (FSK)
    fsk_product1 = np.zeros(1000)
    fsk_product2 = np.zeros(1000)
    fsk_demodulated_signal1 = np.zeros(1000)
    fsk_demodulated_signal2 = np.zeros(1000)
    sequence_demodulated = np.zeros(1000)

    # Умножаем сигналы на опорные волны
    for i in range(0, len(sequence)):
        fsk_product1[i] = sequence[i] * sin(2 * pi * frequency1 * i / 1000)
        fsk_product2[i] = sequence[i] * sin(2 * pi * frequency2 * i / 1000)

    # Выполняем интегрирование и детектирование
    for i in range(0, 10):
        S1 = 0
        S2 = 0
        for t in range(0, 100):
            S1 = S1 + fsk_product1[t + 100 * (i - 1)]
            fsk_demodulated_signal1[t + 100 * (i - 1)] = S1
            S2 = S2 + fsk_product2[t + 100 * (i - 1)]
            fsk_demodulated_signal2[t + 100 * (i - 1)] = S2

    # Применяем пороговую функцию для получения демодулированного сигнала
    ask_demodulated = 1 / 2 * (np.sign(fsk_demodulated_signal1 - fsk_demodulated_signal2) + 1)

    # Восстанавливаем последовательность символов
    for i in range(0, 10):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]

    return fsk_demodulated_signal1, fsk_demodulated_signal2, sequence_demodulated


def create_noise(mean, standard_deviation, length):
    # Создаем шум
    return np.random.normal(mean, standard_deviation, length)


def noise_stress(sequence, sequence_modulated, modulation, frequency):
    error_modulated = []
    sequence_demodulated = np.zeros(1000)
    for i in range(0, 20):
        p = 0
        for m in range(0, 200):
            noise = create_noise(0, 1, 1000)
            sequence_noise = sequence_modulated + i * noise
            if modulation == "ASK":
                ask_demodulated_signal, sequence_demodulated = ask_demodulation(frequency[0], sequence_noise)
            elif modulation == "PSK":
                psk_demodulated_signal, sequence_demodulated = psk_demodulation(frequency[0], sequence_noise)
            elif modulation == "FSK":
                fsk_demodulated_signal1, fsk_demodulated_signal2, sequence_demodulated = fsk_demodulation(frequency[0],
                                                                                                          frequency[1], sequence_noise)
            summa = abs(sum(sequence - sequence_demodulated))
            p += summa / 1000
        error_modulated += [p / 200]
    return error_modulated


def main(ask, psk, fsk1, fsk2):
    sequence = create_sequence()
    x = np.arange(len(sequence)) / 1000
    plot(x, sequence, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Згенерована випадкова послідовність")
    sequence_ask = ask_modulation(ask, sequence)
    plot(x, sequence_ask, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція")
    x_spectrum, spectrum_sequence_ask = spectrum_seq(sequence_ask)
    plot(x_spectrum, spectrum_sequence_ask, axis_x="Частота, Гц", axis_y="Амплітуда спектру",
         title="Спектр при амплітудніймодуляції")
    sequence_psk = psk_modulation(psk, sequence)
    plot(x, sequence_psk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція")
    x_spectrum, spectrum_sequence_psk = spectrum_seq(sequence_psk)
    plot(x_spectrum, spectrum_sequence_psk, axis_x="Частота, Гц", axis_y="Амплітуда спектру",
         title="Спектр при фазовій модуляції")
    sequence_fsk = fsk_modulation(fsk1, fsk2, sequence)
    plot(x, sequence_fsk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція")
    x_spectrum, spectrum_sequence_fsk = spectrum_seq(sequence_fsk)
    plot(x_spectrum, spectrum_sequence_fsk, axis_x="Частота, Гц",
         axis_y="Амплітуда спектру", title="Спектр при частотній модуляціїмодуляції")
    noise = create_noise(0, 1, 1000)
    sequence_ask_noise = sequence_ask + noise
    sequence_psk_noise = sequence_psk + noise
    sequence_fsk_noise = sequence_fsk + noise
    plot(x, sequence_ask_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція з шумом")
    plot(x, sequence_psk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція з шумом")
    plot(x, sequence_fsk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція з шумом")
    ask_demodulated_signal, sequence_demodulated = ask_demodulation(ask, sequence_ask_noise)
    plot(x, ask_demodulated_signal, axis_x="Час, c", axis_y="Амплітудасигналу",
         title="Демодульований сигнал з амплітудною модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з амплітудною модуляцією")
    psk_demodulated_signal, sequence_demodulated = psk_demodulation(psk, sequence_psk_noise)
    plot(x, psk_demodulated_signal, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал з фазовою модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з фазовою модуляцією")
    fsk_demodulated_signal1, fsk_demodulated_signal2, sequence_demodulated = fsk_demodulation(fsk1, fsk2, sequence_fsk_noise)
    plot(x, fsk_demodulated_signal1, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал 1 з частотною модуляцією")
    plot(x, fsk_demodulated_signal2, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал 2 з частотною модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з частотною модуляцією")
    error_ask = noise_stress(sequence, sequence_ask, "ASK", [ask])
    error_psk = noise_stress(sequence, sequence_psk, "PSK", [psk])
    error_fsk = noise_stress(sequence, sequence_fsk, "FSK", [fsk1, fsk2])
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(np.arange(0, 20), error_ask, linewidth=1)
    ax.plot(np.arange(0, 20), error_psk, linewidth=1)
    ax.plot(np.arange(0, 20), error_fsk, linewidth=1)
    ax.set_xlabel('Діапазон змін шуму', fontsize=14)
    ax.set_ylabel('Ймовірність помилки', fontsize=14)
    ax.legend(['ASK', 'PSK', 'FSK'], loc=2)
    plt.title('Оцінка завадостійкості трьох видів модуляції', fontsize=14)
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')
    fig.savefig('./figures/' + 'Оцінка завадостійкості трьох видів модуляції' + '.png', dpi=600)
    plt.close(fig)

main(40, 40, 60, 30)