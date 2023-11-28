import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import gc
import joblib

# Constants that define the stft and segmentation shapes
FRAME_SIZE = 32 #each STFT uses 32 samples
HOP_SIZE = int(FRAME_SIZE/2) #Set hop size to half the frame size (50% overlap)
NFB = int(FRAME_SIZE/2) + 1 #number of frequency bins is equal to 1 more than half the frame size due to DFT symmetry
FRAMES_PER_SEGMENT = 27 #number of frames per spectrotemporal surface
FLOATS_PER_SEGMENT = FRAMES_PER_SEGMENT*NFB #number of data entries in the spectrotemporal surface
SAMPLE_RATE = 22050 #sampling rate is 44.1kHz, but i load them at 22.05kHz (librosa default)
DURATION = FRAME_SIZE * FRAMES_PER_SEGMENT / SAMPLE_RATE # Samples per segment / sample rate = segment duration

wd = os.getcwd()

#read songs
songlist = []
for root, _, files in os.walk(wd + "/Resources/"):
    for file in files:
        if file.endswith(".wav"):
            songlist.append(librosa.load(root + file))
del root
del _
del files
gc.collect()

#compute stfts for each song
spectrograms = []
for song in songlist:
    spectrograms.append(librosa.stft(song[0], n_fft=FRAME_SIZE, hop_length=HOP_SIZE))#song[0] since each song is ([song_data], sample_rate])
#manually free memory
del song
del songlist
gc.collect()

#convert magnitude/amplitude to power
mag_spectrograms = []
for spectrogram in spectrograms:
    mag_spectrograms.append(np.abs(spectrogram) ** 2) #power proportional to square of amplitude
del spectrogram
del spectrograms
gc.collect()

#power to db
mag_spectrograms_db = []
for mag_spectrogram in mag_spectrograms:
    mag_spectrograms_db.append(librosa.power_to_db(mag_spectrogram)) #human perception of sound is logarithmic
del mag_spectrogram
del mag_spectrograms
gc.collect()

#segment the songs
segments = []
for mag_spectrogram_db in mag_spectrograms_db:
    #mag_spectrogram_db has shape (NFB, num_frames in song)
    #rows represent frequency, columns represent time
    #transpose so that rows represent time and columns represent frequency
    #now each row represents one STFT
    mag_spectrogram_db = np.transpose(mag_spectrogram_db)
    #convert spectrogram into 1D array, every NFB is the next time instant (STFT)
    mag_spectrogram_db = mag_spectrogram_db.flatten()

    #every "FLOATS_PER_SEGMENT" is a new segment
    #however, the spectrogram may not have integer number of segments
    #but segment vectors MUST be the same length for PCA, so we should discard the remainder frames
    toRemove = mag_spectrogram_db.shape[0] % FLOATS_PER_SEGMENT #calculate number of excess stfts (frames)
    if toRemove != 0: #remove remainder stfts so that we have an integer number of segments
        mag_spectrogram_db = np.delete(mag_spectrogram_db, np.s_[-toRemove:])
    
    #split the spectrum into segments of fixed length "FLOATS_PER_SEGMENT"
    song_segments = np.array_split(mag_spectrogram_db, len(mag_spectrogram_db) / FLOATS_PER_SEGMENT)
    for segment in song_segments:
        segments.append(segment) #append each segment to the master segment list
del mag_spectrogram_db
del mag_spectrograms_db
del toRemove
gc.collect()

segments = np.array(segments) #convert to numpy array (array where each entry is a vector of length "FLOATS_PER_SEGMENT")
segments = segments.astype("float")
normalized_data = [scale(segment, with_std=False) for segment in segments] #normalize by loudness
del segments
gc.collect()

num_comps = 12 #12 principal components
model = PCA(n_components=num_comps)
model.fit_transform(normalized_data) #perform pca on dataset
joblib.dump(model, 'fitted_pca_model.joblib')

#access principal components
components = model.components_

#plot formatting
num_rows = 3
num_cols = 4
fig, axs = plt.subplots(num_rows, num_cols)
for i in range(num_comps):
    row = i // num_cols
    col = i % num_cols

    component = np.array(components[i])
    #component is row vector of length floats_per_segment
    #to restructure as the original spectrotemporal surface, split the vector every NFB
    #recall that every NFB entries is the next stft
    #this produces an array where rows are time instances (stfts), and columns are frequencies
    #for plotting purposes we want the opposite, so we transpose after reshaping
    component = component.reshape(FRAMES_PER_SEGMENT, NFB).T
    x_vals = np.linspace(0, DURATION, FRAMES_PER_SEGMENT) #linearly space from 0 to duration, with frames_per_segment entries
    y_vals = np.linspace(0, SAMPLE_RATE/2, NFB) #stft has NFB bins and it represents f=[0,SAMPLE_RATE/2] (half sample rate because of dft symmetry)

    X, Y = np.meshgrid(x_vals, y_vals)
    axs[row, col].pcolor(X, Y, component, cmap='hot')
    axs[row, col].set_title(f'Principal Component {i + 1}')
    axs[row, col].set_xlabel('Time (ms)')
    axs[row, col].set_ylabel('Freq (Hz)')

plt.tight_layout()
plt.show()