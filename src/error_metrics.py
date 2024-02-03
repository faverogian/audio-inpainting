import numpy as np
import os
from pydub import AudioSegment
import librosa
from tempfile import mktemp
from . import spectrogram_operations
from . import pca_recomposition


def rmse(M1, M2):
    squared_diff = (M1-M2) ** 2 #matrix difference squared elementwise
    msd = np.mean(squared_diff) #mean squared difference
    rmse = np.sqrt(msd) #root mean squared error
    return rmse

def mae(M1, M2): #mean average error
    return np.mean(np.abs(M1 - M2))

def average_rmse():
    #Avg RMSE
    sum = 0
    idx = 0
    for root, _, files in os.walk("audioAnalysis/data/500_mp3/mp3/test/"):
        for mp3 in files:
            mp3_audio = AudioSegment.from_mp3(root + mp3)
            wname = mktemp('.wav')  # use temporary file
            mp3_audio.export(wname, format="wav")  # convert to wav
            song, _ = librosa.load(wname)
            Log_power, _ = spectrogram_operations.log_power_spectrogram(song)
            reconstructed_spectrogram = pca_recomposition.recompose_spectrogram(song, plot_spectrogram=False)
            Log_power = spectrogram_operations.truncate_spectrogram(Log_power)
            Log_power = spectrogram_operations.array_to_spectrogram_shape(Log_power)

            curErr = rmse(reconstructed_spectrogram, Log_power)
            idx += 1
            print("Error of test " + str(idx) + ": " + str(curErr))
            sum += curErr
    
    avgRmse = sum/idx
    print("Average RMSE: " + str(avgRmse))
    return avgRmse