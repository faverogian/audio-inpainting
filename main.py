import librosa
import spectrogram_operations
import error_metrics
import soundfile as sf
import os
import config_env
import pca_recomposition

#STFT and basis parameterizations
config_env.load_ini_env("environmentVars.ini")
FRAME_SIZE = int(os.environ.get("frame_size"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
NUM_COMPONENTS = int(os.environ.get("num_components"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
HOP_SIZE = int(FRAME_SIZE/2)

#load resource and obtain log power spectrum
file = "path/to/file"
song, _ = librosa.load(file)

#control variable: the log power spectrogram of the original song
Log_power, phase_spectrogram = spectrogram_operations.log_power_spectrogram(song) # keep phase for time recovery
Log_power = spectrogram_operations.truncate_spectrogram(Log_power)
Log_power = spectrogram_operations.array_to_spectrogram_shape(Log_power)
phase_spectrogram = spectrogram_operations.truncate_spectrogram(phase_spectrogram)
phase_spectrogram = spectrogram_operations.array_to_spectrogram_shape(phase_spectrogram)

# spectrogram_operations.plot_spectrogram(Log_power, SAMPLE_RATE, HOP_SIZE, "Log Power Spectrogram")

#Reconstructed spectrogram
reconstructed_spectrogram = pca_recomposition.recompose_spectrogram(song, plot_spectrogram=False)

#Calculate errors
print("RMSE: " + str(error_metrics.rmse(reconstructed_spectrogram, Log_power)))
print("MAE: " + str(error_metrics.mae(reconstructed_spectrogram, Log_power)))

#convert original log power spectrogram to audio
spectrogram = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
original_audio = librosa.istft(spectrogram, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
sf.write(file.split(".")[0] +  "_nc" + str(NUM_COMPONENTS) + "_fs" + str(FRAME_SIZE) + "_FPS" + str(FRAMES_PER_SEGMENT) + "_istft.wav", original_audio, samplerate=SAMPLE_RATE)

#reconstructed audio
reconstructed_spectrogram = spectrogram_operations.LogPowerPhase_to_ComplexSpectrogram(reconstructed_spectrogram, phase_spectrogram)
reconstructed_audio = librosa.istft(reconstructed_spectrogram, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
sf.write(file.split(".")[0] + "_nc" + str(NUM_COMPONENTS) + "_fs" + str(FRAME_SIZE) + "_FPS" + str(FRAMES_PER_SEGMENT) +  "_reconstructed.wav", reconstructed_audio, samplerate=SAMPLE_RATE)
