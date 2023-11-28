import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

FRAME_SIZE = 2048 #each STFT uses this many samples
HOP_SIZE = int(FRAME_SIZE/2) #Set hop size to half the frame size (50% overlap)
SAMPLE_RATE = 22050 #sampling rate is 44.1kHz, but i load them at 22.05kHz (librosa default)

def plot_spectrogram(Y, sr, hop_length, title, y_axis="log"):
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

# if you want to use this script as a standalone script, you might use some code like that shown below
# otherwise you can import it into other files and use it modularly just for the plot_spectrogram function
# song, _ = librosa.load("/audiopath")
# S_song = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
# Song_power = np.abs(S_song) ** 2
# Log_power = librosa.power_to_db(Song_power)
# plot_spectrogram(Log_power, SAMPLE_RATE, HOP_SIZE, title="song")