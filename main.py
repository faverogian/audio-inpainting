import librosa
from src import spectrogram_operations, error_metrics, config_env, pca_recomposition, pca_decomposition
import soundfile as sf
import os

#STFT and basis parameterizations
repo_root = os.path.abspath(os.path.join(__file__, ".."))
env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
config_env.load_ini_env(env)
FRAME_SIZE = int(os.environ.get("frame_size"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
HOP_SIZE = int(FRAME_SIZE/2)


#Things you might want to do


#1. perform pca
# pca_decomposition.generate_pca_bases(plot_bases=True, save_bases_as_audio=True)


#2. visualize bases
spectrogram_operations.plot_pca_bases()


#load resource and obtain log power spectrum
file = repo_root + "/Resources/audiofile.wav"
songname = file.split("/")[-1].split(".")[0]
song, _ = librosa.load(file)


#3. obtain log power spectrum
Log_power, phase_spectrogram = spectrogram_operations.log_power_spectrogram(song) # keep phase for time recovery
spectrogram_operations.plot_spectrogram(Log_power, SAMPLE_RATE, HOP_SIZE, "Log Power Spectrogram")


#4. project the song onto the bases and recompose it to the spectrogram shape
reconstructed_spectrogram = pca_recomposition.recompose_spectrogram(song, plot_spectrogram=True)


#5. reshape the log power spectrum so it can be numerically compared against the reconstruction
Log_power = spectrogram_operations.truncate_spectrogram(Log_power)
Log_power = spectrogram_operations.array_to_spectrogram_shape(Log_power)


#Calculate difference
print("RMSE: " + str(error_metrics.rmse(reconstructed_spectrogram, Log_power)))
print("MAE: " + str(error_metrics.mae(reconstructed_spectrogram, Log_power)))


#6. convert spectrogram back to sound
phase_spectrogram = spectrogram_operations.truncate_spectrogram(phase_spectrogram) #reshape phase spectrogram
phase_spectrogram = spectrogram_operations.array_to_spectrogram_shape(phase_spectrogram)


#convert original log power spectrogram to audio
spectrogram = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
original_audio = librosa.istft(spectrogram, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
sf.write(repo_root + "/Reconstructions/" + #encode environment configuration info in file name
        songname + "_nc" + str(NUM_COMPONENTS) 
        + "_fs" + str(FRAME_SIZE) 
        + "_FPS" + str(FRAMES_PER_SEGMENT) 
        + "_istft.wav", original_audio, samplerate=SAMPLE_RATE)


#reconstructed audio
reconstructed_spectrogram = spectrogram_operations.LogPowerPhase_to_ComplexSpectrogram(reconstructed_spectrogram, phase_spectrogram)
reconstructed_audio = librosa.istft(reconstructed_spectrogram, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
sf.write(repo_root + "/Reconstructions/" + 
        songname + "_nc" + str(NUM_COMPONENTS) 
        + "_fs" + str(FRAME_SIZE) 
        + "_FPS" + str(FRAMES_PER_SEGMENT) 
        +  "_reconstructed.wav", reconstructed_audio, samplerate=SAMPLE_RATE)
