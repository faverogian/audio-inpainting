import joblib
import numpy as np
import librosa
import spectrogram
import error
from sklearn.preprocessing import scale

#STFT and basis parameterizations
FRAME_SIZE = 32
HOP_SIZE = int(FRAME_SIZE/2)
NFB = int(FRAME_SIZE/2) + 1
FRAMES_PER_SEGMENT = 27
FLOATS_PER_SEGMENT = FRAMES_PER_SEGMENT*NFB
SAMPLE_RATE = 22050
DURATION = FRAME_SIZE * FRAMES_PER_SEGMENT / SAMPLE_RATE

#load model
model = joblib.load('fitted_pca_model.joblib')
components = model.components_
num_comps = components.shape[0]

#load resource and obtain log power spectrum
song, _ = librosa.load("Resources/swan.wav")
S_song = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Song_power = np.abs(S_song) ** 2
Log_power = librosa.power_to_db(Song_power)

#plot log power spectrogram
# spectrogram.plot_spectrogram(Log_power, SAMPLE_RATE, HOP_SIZE, "Original")

#preprocessing
Log_power_T = Log_power.T #convert from frequency rows and column time to time rows and frequency columns
Log_power_T = Log_power_T.flatten() #make one dimensional
toRemove = Log_power_T.shape[0] % FLOATS_PER_SEGMENT #prepare to remove excess samples
if toRemove != 0:
    Log_power_T = np.delete(Log_power_T, np.s_[-toRemove:])
segments = np.array_split(Log_power_T, len(Log_power_T) / FLOATS_PER_SEGMENT)

#for each segment we need to scale (shift by average loudness to be zero mean)
scaled_segments = []
for segment in segments:
    segment = segment.astype('float')
    scaled_segment = scale(segment, with_std=False) #keep the spread/dispersion of loudness
    scaled_segments.append(scaled_segment)

# calculate projections
projections = []
for scaled_segment in scaled_segments:
    projection = np.zeros(num_comps)
    for i in range(num_comps): #projection onto each basis
        projection[i] = (np.dot(scaled_segment, components[i])/np.dot(components[i], components[i]))
    projections.append(projection)

# compute linear combination of projections
reconstructed = []
for projection in projections:
    reconstructed_segment = np.zeros(FLOATS_PER_SEGMENT)
    for i in range(num_comps):
        reconstructed_segment = reconstructed_segment + components[i]*projection[i]
    reconstructed.append(reconstructed_segment)

# reshape to plot scaled input
scaled_segments = np.array(scaled_segments).flatten()#convert to array and flatten
num_samples = scaled_segments.shape[0] #total number of floats (bins)
scaled_segments = scaled_segments.reshape(int(num_samples/NFB), NFB).T#convert to 2D array where rows are frequency and columns are time

#reshape reconstruction to spectrogram dimensions
reconstructed = np.array(reconstructed)
reconstructed = reconstructed.flatten()
reconstructed = reconstructed.reshape(int(num_samples/NFB), NFB).T

# compute difference (reconstruction error)
print("RMSE: " + str(error.rmse(reconstructed, scaled_segments)))
print("MAE: " + str(error.mae(reconstructed, scaled_segments)))

# plot original and reconstruction
spectrogram.plot_spectrogram(scaled_segments, SAMPLE_RATE, HOP_SIZE, "Scaled Original")
spectrogram.plot_spectrogram(reconstructed, SAMPLE_RATE, HOP_SIZE, "Reconstructed")