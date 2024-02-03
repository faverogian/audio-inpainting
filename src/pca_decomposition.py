import os
import librosa
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import joblib
import math
from . import config_env, spectrogram_operations
from tempfile import mktemp
from pydub import AudioSegment


repo_root = os.path.abspath(os.path.join(__file__, "../.."))
env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
config_env.load_ini_env(env)
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAME_SIZE = int(os.environ.get("frame_size")) #samples per stft
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment")) #frames per segment 

# a segment is a time denomination, or interval, of ~50ms for best results.
# segment duration = Frame_Size * Frames_per_Segment / Sample_Rate

def generate_pca_bases(plot_bases=False, save_bases_as_audio=False):
    #read songs
    ipca = IncrementalPCA(n_components=NUM_COMPONENTS, batch_size=1000)
    normalized_data = np.empty((0,459)) #collection of all 50ms intervals
    count = 0
    for root, _, files in os.walk(repo_root + "/data/500_mp3/mp3/train/"):
        for file in files:
            if file.endswith(".mp3"):
                count += 1
                print(str(count) + "/" + str(len(files)))
                mp3_audio = AudioSegment.from_mp3(root + file)
                wname = mktemp('.wav')  # use temporary file
                mp3_audio.export(wname, format="wav")  # convert to wav
                song, _ = librosa.load(wname)
                power_spectrogram, _ = spectrogram_operations.log_power_spectrogram(song) #we just want magnitude bases
                truncated_spectrogram = spectrogram_operations.truncate_spectrogram(power_spectrogram) #fit data to schema
                segmented_spectrogram = spectrogram_operations.segment_spectrogram(truncated_spectrogram) #cut into segments
                for segment in segmented_spectrogram:
                    normalized_segment = spectrogram_operations.normalize_segment(segment)[0].reshape(1,-1)
                    normalized_data = np.append(normalized_data, normalized_segment, axis=0)
                    if normalized_data.shape[0] == 1000:
                        ipca.partial_fit(normalized_data)
                        normalized_data = np.empty((0,459)) #collection of all 50ms intervals

    print("segments")

    # model = PCA(n_components=NUM_COMPONENTS) #perform pca on dataset
    # print("model")
    # model.fit_transform(normalized_data)
    # print("transformed")

    joblib.dump(ipca, repo_root + '/models/ipca_model_nc' + str(NUM_COMPONENTS) #save model and include parameters in title
                + '_fs' + str(FRAME_SIZE) 
                + "_FPS" + str(FRAMES_PER_SEGMENT) 
                + ".joblib")

    # Visualization
    if(plot_bases == True or save_bases_as_audio == True):

        components = ipca.components_
        
        if(plot_bases == True):
            #plot on a roughly square grid
            num_cols = math.ceil(math.sqrt(NUM_COMPONENTS)) #take square root and round up
            num_rows = math.ceil(NUM_COMPONENTS/num_cols) #find the minimum the other factor must be
            spectrogram_operations.plot_pca_bases(num_rows, num_cols, components)

        if(save_bases_as_audio == True):#save pca bases as audio files
            spectrogram_operations.pca_bases_to_audio(components)