import matplotlib.pyplot as plt
import sounddevice as sd

from BPSK import BPSK

bpsk = BPSK()

# Get 4 integers from the console
try:
    bpsk.fs = int(input("Unesite stopu uzorkovanja (veće od 0): "))
    bpsk.baud = int(input("Unesite stopu simbola (veće od 0): "))
    bpsk.nbits = int(input("Unesite broj bitova (veći od 0): "))
    bpsk.f0 = int(input("Unesite frekvencu nosača (veću od 0): "))
    bpsk.noise_amplitude = float(input("Unesite nivo šuma (0.1 = 10%): "))

    # Check if all numbers are above 0
    if bpsk.fs > 0 and bpsk.baud > 0 and bpsk.nbits > 0 and bpsk.f0 > 0:
        bpsk.calculate_number_of_samples_per_symbol()
        bpsk.calculate_number_of_samples()
        bpsk.set_up_time()
        bpsk.set_up_visible_limit()
        bpsk.make_bpsk_signal_with_noise()

    else:
        print("Please enter integers above 0 for all inputs.")
        exit(1)
except ValueError as e:
    print(e)
    exit(1)

# ---------- Plot of BPSK ------------#
# ---------- Time Domain Signals ----------#
fig, axis = plt.subplots(3, 1)
plt.title('BPSK Modulation', fontsize=12)

axis[0].plot(bpsk.t, bpsk.input_signal, color='C1')
axis[0].set_title('Input Signal')
axis[0].set_xlabel('Time [s]')
axis[0].set_xlim(0, bpsk.time_domain_visible_limit)
axis[0].set_ylabel('Amplitude [V]')
axis[0].grid(linestyle='dotted')

axis[1].plot(bpsk.t, bpsk.carrier_signal, color='C2')
axis[1].set_title('Carrier Signal')
axis[1].set_xlabel('Time [s]')
axis[1].set_xlim(0, bpsk.time_domain_visible_limit)
axis[1].set_ylabel('Amplitude [V]')
axis[1].grid(linestyle='dotted')

axis[2].plot(bpsk.t, bpsk.bpsk_signal, color='C3')
axis[2].set_title('BPSK Modulated Signal')
axis[2].set_xlabel('Time [s]')
axis[2].set_xlim(0, bpsk.time_domain_visible_limit)
axis[2].set_ylabel('Amplitude [V]')
axis[2].grid(linestyle='dotted')

plt.suptitle('BPSK Modulation', fontsize=12)
plt.subplots_adjust(hspace=1.3)
plt.show()

ber = bpsk.calculate_ber()

print("\nBER vrednost: {:.4f}".format(ber))
print("BER: {:.4f}%".format(ber * 100))

# sd.play(bpsk.bpsk_signal, bpsk.fs/3)
# sd.wait()
