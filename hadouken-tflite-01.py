# You'll probably need PortAudio:
#   sudo apt install libportaudio2

import sounddevice as sd
import numpy as np 
import timeit
from scipy import signal
from tflite_runtime.interpreter import Interpreter
import math

# Settings
model_path = 'hadouken_model_03.tflite'
labels = ['hadouken', 'other', 'silence']
sample_time = 1.0       # Time for 1 sample (sec)
sample_rate = 48000     # Sample rate (Hz) of microphone
resample_rate = 8000    # Downsample to this rate (Hz)
filter_cutoff = 4000    # Remove frequencies above this threshold (Hz)
num_channels = 1
stft_n_fft = 512        # Number of FFT bins (also, number of samples in each slice)
stft_n_hop = 400        # Distance between start of each FFT slice (number of samples)
stft_window = 'hanning' # "The window of choice if you don't have any better ideas"
stft_min_bin = 1        # Lowest bin to use (inclusive; basically, filter out DC)       
stft_avg_bins = 8       # Number of bins to average together to reduce FFT size
shift_n_bits = 3        # Number of bits to shift 16-bit STFT values to make 8-bit values (before clipping)
ffts_per_inference = 2  # Number of FFTs to compute before performing inference
rec_duration = 0.5

# Calculated parameters
stft_n_slices = int(math.ceil(((sample_time * resample_rate) / stft_n_hop) - 
                (stft_n_fft / stft_n_hop)) + 1)
stft_max_bin = int((stft_n_fft / 2) / ((resample_rate / 2) / filter_cutoff)) + 1
stft_n_bins = (stft_max_bin - stft_min_bin) // stft_avg_bins
stft_n_overlap = stft_n_fft - stft_n_hop
hann_window = np.hanning(stft_n_fft)
print('N slices:', stft_n_slices)
print('STFT max bin:', stft_max_bin)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Resample
def resample(sig, old_fs, new_fs):
    seconds = len(sig) / old_fs
    num_samples = seconds * new_fs
    resampled_signal = signal.resample(sig, int(num_samples))
    #signal = signal / np.iinfo(np.int16).max
    return resampled_signal

# Extract STFT features from window
def extract_stft(sample):
    
    # Convert floating point wav data (-1.0 to 1.0) to 16-bit PCM
    waveform = np.around(sample * 32767)
    
    # Construct STFT manually with real-valued FFT
    hann_window = np.hanning(stft_n_fft)
    stft = np.zeros(((stft_max_bin - stft_min_bin) // stft_avg_bins, stft_n_slices))
    for i in range(stft.shape[1]):

        # Get FFT for window
        win_start = i * stft_n_hop
        win_stop = (i * stft_n_hop) + stft_n_fft

        ### TEST--we get non-conforming waveforms sometimes
        if hann_window.shape != waveform[win_start:win_stop].shape:
            return stft
        
        # Window
        window = hann_window * waveform[win_start:win_stop]
        fft = np.abs(np.fft.rfft(window, n=stft_n_fft))

        # Only keep the frequency bins we care about (i.e. filter out unwanted frequencies)
        fft = fft[stft_min_bin:stft_max_bin]

        # Adjust for quantization and scaling in 16-bit fixed point FFT
        fft = np.around(fft / stft_n_fft)

        # Average every <stft_avg_bins bins> together to reduce size of FFT
        fft = np.mean(fft.reshape(-1, stft_avg_bins), axis=1)

        # Reduce precision by converting to 8-bit unsigned values [0..255]
        fft = np.around(fft / (2 ** shift_n_bits))
        fft = np.clip(fft, a_min=0, a_max=255)

        # Put FFT slice into STFT
        stft[:,i] = fft
    
    return stft

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec = resample(rec, sample_rate, resample_rate)

    print(rec[0:10])
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec
    
    # Extract features
    start = timeit.default_timer()
    stft = extract_stft(window)

    # For testing time
    print('Time (ms):', timeit.default_timer() - start)
    
    # Reshape features
    in_tensor = np.float32(stft.reshape(1,stft.shape[0], stft.shape[1]))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    
    # Infer!
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    
    # Print out result
    #print(val)
    if val >= 0.5:
        print("HADOUKEN!")
    #print(labels[np.argmax(output_tensor)])

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass