import spectrogram_operations
import joblib
import spectrogram_operations
import os
import config_env

#STFT and basis parameterizations
config_env.load_ini_env("environmentVars.ini")
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAME_SIZE = int(os.environ.get("frame_size"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))

model = joblib.load('models/pca_model_nc' + str(NUM_COMPONENTS) #load current model
                    + '_fs' + str(FRAME_SIZE) 
                    + "_FPS" + str(FRAMES_PER_SEGMENT) 
                    + ".joblib")
components = model.components_
spectrogram_operations.plot_pca_bases(3,4,components)