from src import config_env
import os
import numpy as np
import joblib

#STFT and basis parameterizations
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

# print(model.explained_variance_ratio_)
print(np.sum(model.explained_variance_ratio_))

