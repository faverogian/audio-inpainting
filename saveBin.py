import librosa
from src import config_env, pca_recomposition
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

#load resource and obtain log power spectrum
file = repo_root + "/Resources/samplesong.wav"
songname = file.split("/")[-1].split(".")[0]
song, _ = librosa.load(file)

binaryFileName = (songname + "_nc" + str(NUM_COMPONENTS) 
        + "_fs" + str(FRAME_SIZE) 
        + "_FPS" + str(FRAMES_PER_SEGMENT) 
        +  "_reconstructed.bin"
    )

reconstructed_spectrogram = pca_recomposition.recompose_spectrogram(song, plot_spectrogram=False, saveBinary=True, binaryFileName=binaryFileName)
