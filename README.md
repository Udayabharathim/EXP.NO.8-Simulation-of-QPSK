# EXP.NO.8-Simulation-of-QPSK



# AIM
imulation of QPSK using Python
# SOFTWARE REQUIRED
Google Collab
# ALGORITHMS
1. Generate a random bit sequence for QPSK modulation (two bits per symbol).
2. Map the bit pairs to QPSK symbols (0, 1, 2, 3) using phase shifts.
3. Initialize the QPSK signal and define phase angles for each symbol.
4. For each symbol, generate a QPSK signal segment with the corresponding phase.
5. Concatenate the signal segments to form the complete QPSK signal.
6. Plot the real and imaginary components of the signal along with the symbol markers.
# PROGRAM
```
import numpy as np 
import matplotlib.pyplot as plt 
# Parameters 
num_symbols = 10 # Number of symbols (reduced for clarity in the plot) 
T = 1.0 # Symbol period 
fs = 100.0 # Sampling frequency 
t = np.arange(0, T, 1/fs) # Time vector for one symbol 
# Generate random bit sequence 
bits = np.random.randint(0, 2, num_symbols * 2) # Two bits per QPSK symbol 
symbols = 2 * bits[0::2] + bits[1::2] # Map bits to QPSK symbols 
# Initialize QPSK signal 
qpsk_signal = np.array([]) 
symbol_times = [] 
# Define the QPSK modulation and phase angles 
symbol_phases = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2} 
# Generate QPSK signal 
for i, symbol in enumerate(symbols): 
phase = symbol_phases[symbol] 
symbol_time = i * T 
qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase) 
qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment)) 
symbol_times.append(symbol_time) 
# Time vector for the entire signal 
t_total = np.arange(0, num_symbols * T, 1/fs) 
# Plot real and imaginary parts of the QPSK signal 
plt.figure(figsize=(14, 12)) 
# Plot the in-phase component with symbols 
plt.subplot(3, 1, 1) 
plt.plot(t_total, np.real(qpsk_signal), label='In-phase') 
for i, symbol_time in enumerate(symbol_times): 
plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5) 
52 
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue') 
plt.title('QPSK Signal - In-phase Component with Symbols') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.legend() 
# Plot the quadrature component with symbols 
plt.subplot(3, 1, 2) 
plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature', color='orange') 
for i, symbol_time in enumerate(symbol_times): 
plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5) 
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue') 
plt.title('QPSK Signal - Quadrature Component with Symbols') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.legend() 
# Plot the resultant QPSK waveform (real part) 
plt.subplot(3, 1, 3) 
plt.plot(t_total, np.real(qpsk_signal), label='Resultant QPSK Waveform', color='green') 
for i, symbol_time in enumerate(symbol_times): 
plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5) 
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue') 
plt.title('Resultant QPSK Waveform') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.legend() 
plt.tight_layout() 
plt.show()
```
# OUTPUT
 ![image](https://github.com/user-attachments/assets/067b2b45-4234-41cc-a8c8-eeada664f47e)

# RESULT / CONCLUSIONS
Thus QPSK modulation is implemented using Scilab code.
