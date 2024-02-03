import librosa
from src import spectrogram_operations, config_env, error_metrics
from src.pca_recomposition import projection, linearCombination
import os
import joblib
import numpy as np
import soundfile as sf
from src import filters

repo_root = os.path.abspath(os.path.join(__file__, ".."))
env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
config_env.load_ini_env(env)
FRAME_SIZE = int(os.environ.get("frame_size"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
HOP_SIZE = int(FRAME_SIZE/2)

model = joblib.load(repo_root + '/models/ipca_model_nc' + str(NUM_COMPONENTS) #load current model
                            + '_fs' + str(FRAME_SIZE) 
                            + "_FPS" + str(FRAMES_PER_SEGMENT) 
                            + ".joblib")

components = model.components_

songname = "/viva-la-vida.wav"
pathToSong = repo_root + songname
song, _ = librosa.load(pathToSong)
Log_power, phase_spectrogram = spectrogram_operations.log_power_spectrogram(song)
Log_power = spectrogram_operations.truncate_spectrogram(Log_power) #truncate to be integer number of segments
segmented_spectrogram = spectrogram_operations.segment_spectrogram(Log_power)
scaled_segments, loudnesses = spectrogram_operations.normalize(segmented_spectrogram)

segmentProjections = []
for scaled_segment in scaled_segments:
    projectionOnBases = projection(NUM_COMPONENTS, scaled_segment, components)
    segmentProjections.append(projectionOnBases)
    reconstructed = []

for index, segmentProjection in enumerate(segmentProjections):
    reconstructed.append(linearCombination(
        NUM_COMPONENTS, 
        segmentProjection, 
        components, 
        loudnesses[index])
    )

Log_power = spectrogram_operations.truncate_spectrogram(Log_power)
Log_power = spectrogram_operations.array_to_spectrogram_shape(Log_power)

reconstructed_spectrogram = spectrogram_operations.array_to_spectrogram_shape(np.array(reconstructed))
print("No LPF RMSE: " + str(error_metrics.rmse(reconstructed_spectrogram, Log_power)))

phase_spectrogram = spectrogram_operations.truncate_spectrogram(phase_spectrogram) #reshape phase spectrogram
phase_spectrogram = spectrogram_operations.array_to_spectrogram_shape(phase_spectrogram)
reconstructed_spectrogram = spectrogram_operations.LogPowerPhase_to_ComplexSpectrogram(reconstructed_spectrogram, phase_spectrogram)
reconstructed_audio = librosa.istft(reconstructed_spectrogram, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
reconstructed_audio = filters.butter_lowpass_filter(reconstructed_audio)
S_song = librosa.stft(reconstructed_audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE) #stft
Song_power = np.abs(S_song) ** 2 #power is proportional to amplitude squared
LPF_Log_power = librosa.power_to_db(Song_power) #human perception of loudness is logarithmic
print("LPF RMSE: " + str(error_metrics.rmse(LPF_Log_power, Log_power)))
