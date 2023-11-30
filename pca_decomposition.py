import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
import joblib
import math
import config_env
import spectrogram_operations

config_env.load_ini_env("environmentVars.ini")
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAME_SIZE = int(os.environ.get("frame_size")) #samples per stft
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment")) #frames per segment 

# a segment is a time denomination, or interval, of ~50ms for best results.
# segment duration = Frame_Size * Frames_per_Segment / Sample_Rate

wd = os.getcwd()

def generate_pca_bases(plot_bases=True, save_bases_as_audio=True):
    #read songs
    songlist = []
    for root, _, files in os.walk(wd + "/Resources/"):
        for file in files:
            if file.endswith(".wav"):
                song, _ = librosa.load(root + file)
                songlist.append(song)

    # compute log power spectrogram for all data, then divide into segments 
    segments = [] #collection of all 50ms intervals
    for song in songlist:
        power_spectrogram, _ = spectrogram_operations.log_power_spectrogram(song) #we just want magnitude bases
        truncated_spectrogram = spectrogram_operations.truncate_spectrogram(power_spectrogram) #fit data to schema
        segmented_spectrogram = spectrogram_operations.segment_spectrogram(truncated_spectrogram) #cut into segments
        for segment in segmented_spectrogram:
            segments.append(segment) #append each segment to the master segment list

    segments = np.array(segments)
    segments = segments.astype("float")
    normalized_data = spectrogram_operations.normalize(segments)[0] #normalize by loudness

    model = PCA(n_components=NUM_COMPONENTS) #perform pca on dataset
    model.fit_transform(normalized_data)

    joblib.dump(model, 'models/pca_model_nc' + str(NUM_COMPONENTS) #save model and include parameters in title
                + '_fs' + str(FRAME_SIZE) 
                + "_FPS" + str(FRAMES_PER_SEGMENT) 
                + ".joblib")

    # Visualization
    if(plot_bases == True or save_bases_as_audio == True):

        components = model.components_
        
        if(plot_bases == True):
            #plot on a roughly square grid
            num_cols = math.ceil(math.sqrt(NUM_COMPONENTS)) #take square root and round up
            num_rows = math.ceil(NUM_COMPONENTS/num_cols) #find the minimum the other factor must be
            spectrogram_operations.plot_pca_bases(num_rows, num_cols, components)

        if(save_bases_as_audio == True):#save pca bases as audio files
            spectrogram_operations.pca_bases_to_audio(components)

generate_pca_bases()