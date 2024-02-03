import librosa
from src import spectrogram_operations, error_metrics, config_env, pca_recomposition, pca_decomposition, partition
import soundfile as sf
import os
from tempfile import mktemp
from pydub import AudioSegment

#STFT and basis parameterizations
# repo_root = os.path.abspath(os.path.join(__file__, ".."))
# env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
# config_env.load_ini_env(env)
# FRAME_SIZE = int(os.environ.get("frame_size"))
# SAMPLE_RATE = int(os.environ.get("sample_rate"))
# NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
# FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
# HOP_SIZE = int(FRAME_SIZE/2)


#Partition (if not yet partitioned)
# partition.partition()


#1. perform pca
pca_decomposition.generate_pca_bases(plot_bases=False, save_bases_as_audio=False)


#2. visualize bases
# spectrogram_operations.plot_pca_bases()

#Avg RMSE
error_metrics.average_rmse()


