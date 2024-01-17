import math

import numpy as np

from numpy import pi
from numpy import cos
from numpy import r_
import matplotlib.pyplot as plt


def GetBpskSymbol(bit1: bool):
    if ~bit1:
        return 0
    elif bit1:
        return 1
    else:
        return -1


class BPSK:
    fs = 0  # Stopa uzorkovanja (sampling rate)
    baud = 0  # Stopa simbola (symbol rate)
    nbits = 0  # Broj bitova (number of bits)
    f0 = 0  # Frekvencija nosioca (carrier frequency)
    Ns = 0  # Broj uzoraka po simbolu (number of Samples per Symbol)
    N = 0  # Ukupan broj uzoraka (Total Number of Samples)
    t = 0  # Tačke vremena (time points)
    noise_amplitude = 0.0  # Amplituda šuma

    # Limit za prikaz vremenskog signala radi bolje vidljivosti.
    symbols_to_show = 25
    time_domain_visible_limit = 0

    carrier_signal = 0  # Nosilac (carrier)
    input_bits = 0  # Ulazni bitovi (input bits)
    input_signal = 0  # Ulazni signal (input signal)
    data_symbols = 0  # Simboli podataka (data symbols)
    bpsk_signal = 0  # BPSK signal

    # Konstruktor metoda (init metoda)
    def __init__(self):
        pass

    # Metoda instance za izračunavanje broja uzoraka po simbolu
    def calculateNumberOfSamplesPerSymbol(self):
        self.Ns = int(self.fs / self.baud)

    # Metoda instance za izračunavanje ukupnog broja uzoraka
    def calculateNumberOfSamples(self):
        self.N = self.nbits * self.Ns

    # Metoda instance za postavljanje vremena
    def setUpTime(self):
        self.t = r_[0.0:self.N] / self.fs

    # Metoda instance za postavljanje vidljivog ograničenja u vremenskom domenu
    def setUpVisibleLimit(self):
        self.time_domain_visible_limit = np.minimum(self.nbits / self.baud, self.symbols_to_show / self.baud)

    # Metoda instance za izračunavanje nosioca
    def calculateCarrier(self):
        self.carrier_signal = cos(2 * pi * self.f0 * self.t)

    # Metoda instance za dodeljivanje ulaznih bitova
    def assignInputBits(self):
        self.input_bits = np.random.randn(self.nbits, 1) > 0

    def calculateNoise(self):
        return self.noise_amplitude * np.random.normal(size=len(self.t))

    # Metoda instance za digitalno-analognu konverziju
    def digitalToAnalogConvertor(self):
        self.input_signal = (np.tile(self.input_bits * 2 - 1, (1, self.Ns))).ravel()
        self.data_symbols = np.array([[GetBpskSymbol(self.input_bits[x])] for x in range(0, self.input_bits.size)])

    def calculateBSPKSignalWithNoise(self):
        self.calculateCarrier()
        self.assignInputBits()
        self.digitalToAnalogConvertor()

        # Generisanje Gausovog šuma
        noise = self.calculateNoise()
        input_signal_with_noise = self.input_signal + noise

        # Dodavanje šuma na BPSK signal
        self.bpsk_signal = input_signal_with_noise * self.carrier_signal

    def calculateBER(self):
        # Demodulacija BPSK signala
        demodulated_signal = self.bpsk_signal * self.carrier_signal

        # Filtriranje signala
        # Bolji BER se dobija bez filtriranja signala
        # filtrirani_signal = np.convolve(demodulated_signal, np.ones(self.Ns) / self.Ns, mode='valid')

        # Vizualizacija filtriranog signala
        # plt.plot(self.t[:len(filtrirani_signal)], filtrirani_signal)
        # plt.show()

        # Računanje praga odlučivanja kao srednje vrednosti između min i max vrednosti filtriranog signala
        # Za bolji rezultat stavljen je na 0
        decision_threshold = 0

        # plt.plot(demodulated_signal, label='Demodulisani Signal')
        # plt.axhline(y=decision_threshold, color='r', linestyle='--', label='Prag Odlučivanja')
        # plt.xlabel('Vreme')
        # plt.ylabel('Amplituda')
        # plt.title('Demodulisani Signal i Prag Odlučivanja')
        # plt.legend()
        # plt.show()

        # Odbiranje i donošenje odluka sa pravilnim pragom
        received_bits = (demodulated_signal > decision_threshold).astype(int)

        # Računanje grešaka
        errors = 0

        it_start = 0
        it_end = 39
        for input_bit in self.input_bits:
            # Upoređivanje sa ulaznim bitovima
            if not (input_bit != np.all(received_bits[it_start:it_end] == 0)):
                errors += 1
            it_start += 40
            it_end += 40

        # Računanje BER
        if errors == 0:
            return 0
        else:
            ber = errors / self.nbits
            return ber

        # Dodatni ispis
        # print("Filtrirani Signal:", filtrirani_signal)
        # print("Prag Odlučivanja:", decision_threshold)
        # print("Primljeni Bitovi:", received_bits)
        # print("Ulazni Bitovi:", self.ulazni_bitovi)
        # print("Min Filtrirani Signal:", np.min(filtrirani_signal))
        # print("Max Filtrirani Signal:", np.max(filtrirani_signal))
