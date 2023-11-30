import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import config_env
from sklearn.preprocessing import scale
import spectrogram_operations
import soundfile as sf

config_env.load_ini_env("environmentVars.ini")
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAME_SIZE = int(os.environ.get("frame_size"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
HOP_SIZE = int(FRAME_SIZE/2)
NFB = int(FRAME_SIZE/2) + 1 #number of frequency bins is equal to 1 more than half the frame size due to DFT symmetry
FLOATS_PER_SEGMENT = FRAMES_PER_SEGMENT*NFB #number of data entries in the spectrotemporal surface
DURATION = FRAME_SIZE * FRAMES_PER_SEGMENT / SAMPLE_RATE # Samples per segment / sample rate = segment duration

def plot_spectrogram(Y, sr, hop_length, title, y_axis="log"):
    #y_axis defaults to log since frequency perception is logarithmic base 2 (1 octave = frequency doubling)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis,
                             cmap='inferno',
                             )
    plt.title(title)
    plt.ylim(bottom=0)
    plt.colorbar(format="%+2.f")
    plt.show()

def log_power_spectrogram(song):
    S_song = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE) #stft
    phase = np.angle(S_song)
    Song_power = np.abs(S_song) ** 2 #power is proportional to amplitude squared
    Log_power = librosa.power_to_db(Song_power) #human perception of loudness is logarithmic
    return (Log_power, phase)

def log_power_to_stft(Log_power):
    Song_power = librosa.db_to_power(Log_power) #db back to power
    S_song = np.sqrt(Song_power) #power to amplitude
    return S_song #this is the original stft

def round_trip(song): #perform stft bring to log power, then back to time signal
    return librosa.istft(log_power_to_stft(log_power_spectrogram(song)), hop_length=HOP_SIZE, n_fft=FRAME_SIZE)

def truncate_spectrogram(spectrogram):
    # segment vectors MUST be equal length for PCA discard excess frames (will be less than 1 segment length: ~50ms)

    # rows represent frequency, columns represent time
    # transpose so that rows are time and cols are freq bins
    spectrogram = spectrogram.T # each row represents one STFT
    spectrogram = spectrogram.flatten()

    toRemove = len(spectrogram) % FLOATS_PER_SEGMENT #calculate number of excess stfts (frames)
    if toRemove != 0: #remove remainder stfts so that we have an integer number of segments
        spectrogram = np.delete(spectrogram, np.s_[-toRemove:])
    return spectrogram

def segment_spectrogram(truncated_spectrogram):
    #split the spectrogram into segments of fixed length "FLOATS_PER_SEGMENT"
    segmented_spectrogram = np.array_split(truncated_spectrogram, len(truncated_spectrogram) / FLOATS_PER_SEGMENT)
    return segmented_spectrogram

def normalize(segmented_spectrogram):
    loudnesses = []
    scaled_segments = []
    for segment in segmented_spectrogram:
        segment = segment.astype('float')
        loudness = np.mean(segment)
        loudnesses.append(loudness)
        scaled_segment = scale(segment, with_std=False) #keep the spread/dispersion of loudness
        scaled_segments.append(scaled_segment)
    return (scaled_segments, loudnesses)

def array_to_spectrogram_shape(arr):
    arr = arr.flatten()

    #recall that every NFB entries is the next stft, .reshape(num_samples/NFB, NFB) preserves order 
    #and produces an array where rows are time instances (stfts), and columns are frequencies
    #for plotting purposes we want the opposite, so we transpose after reshaping

    num_samples = len(arr)
    reshaped_arr = arr.reshape(int(num_samples/NFB), NFB).T
    return reshaped_arr

def LogPowerPhase_to_ComplexSpectrogram(log_power_spectrogram, phase_spectrogram):
    #combine magnitude and phase information
    mag_spectrogram = spectrogram_operations.log_power_to_stft(log_power_spectrogram)
    recover_complex_rectangular_form = np.vectorize(lambda mag, phi: complex(mag * np.cos(phi), mag * np.sin(phi)))
    reconstructed_spectrogram = recover_complex_rectangular_form(mag_spectrogram, phase_spectrogram)
    return reconstructed_spectrogram

# def LogPowerPhaseSpectrogram_to_audio(log_power_spectrogram, phase_spectrogram):
#     complex_spectrogram = LogPowerPhase_to_ComplexSpectrogram(log_power_spectrogram, phase_spectrogram)
#     audio = librosa.istft(complex_spectrogram, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)
#     return audio

def plot_pca_bases(num_rows, num_cols, components):
    _, axs = plt.subplots(num_rows, num_cols)

    for i in range(len(components)):
        row = i // num_cols
        col = i % num_cols

        component = np.array(components[i])
        component = array_to_spectrogram_shape(component)

        #linearly space from 0 to duration with frames_per_segment number of divisions
        x_vals = np.linspace(0, DURATION, FRAMES_PER_SEGMENT)
        #stft has NFB bins and it represents f=[0,SAMPLE_RATE/2] (half sample rate because of dft symmetry)
        y_vals = np.linspace(0, SAMPLE_RATE/2, NFB) 
        X, Y = np.meshgrid(x_vals, y_vals)
        axs[row, col].pcolor(X, Y, component, cmap='hot')
        axs[row, col].set_title(f'Principal Component {i + 1}')
        axs[row, col].set_xlabel('Time (ms)')
        axs[row, col].set_ylabel('Freq (Hz)')

    plt.tight_layout()
    plt.show()

def pca_bases_to_audio(components):
    folder = "bases/pca_model_nc" + str(NUM_COMPONENTS) + "_fs" + str(FRAME_SIZE) + "_FPS" + str(FRAMES_PER_SEGMENT) +"/"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for i, component in enumerate(components):
        component = np.array(component)
        component = array_to_spectrogram_shape(component)

        pca_i = spectrogram_operations.log_power_to_stft(component)
        pca_i_audio = librosa.istft(pca_i, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)
        sf.write(folder + "component_"  + str(i) + ".wav", pca_i_audio, samplerate=SAMPLE_RATE)

def amplify_audio(song):
    return 10*song