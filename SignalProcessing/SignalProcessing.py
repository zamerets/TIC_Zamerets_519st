import numpy
import scipy
import matplotlib.pyplot as plt
from scipy import signal, fft

# вариант 10 Замерець

n = 500
Fs = 1000
Fmax = 21


random = numpy.random.normal(0, 10, n)
print(random)
time_line_ox = numpy.arange(n)/Fs
print(time_line_ox)

w = Fmax/(Fs/2)

parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')



filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, random)

print(filtered_signal)

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
ax.set_xlabel("Время (секунды) ", fontsize = 14)
ax.set_ylabel("Амплитуда спектра ", fontsize = 14)

plt.title("Спектр сигнала с максимальной частотой Fmax = 21", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 2' + '.png', dpi=600)