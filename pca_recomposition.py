import joblib
import numpy as np
import spectrogram_operations
import os
import config_env

#STFT and basis parameterizations
config_env.load_ini_env("environmentVars.ini")
NUM_COMPONENTS = int(os.environ.get("num_pca_components"))
FRAME_SIZE = int(os.environ.get("frame_size"))
FRAMES_PER_SEGMENT = int(os.environ.get("frames_per_segment"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))#audio sampling rate
HOP_SIZE = int(FRAME_SIZE/2) #Set hop size to half the frame size (50% overlap)

model = joblib.load('models/pca_model_nc' + str(NUM_COMPONENTS) #load current model
                    + '_fs' + str(FRAME_SIZE) 
                    + "_FPS" + str(FRAMES_PER_SEGMENT) 
                    + ".joblib")
components = model.components_

def projection(num_components, segment_vector, bases):
    projection = np.zeros(num_components)
    for i in range(num_components): #iterate through bases and compute projection
        projection[i] = (np.dot(segment_vector, bases[i])/np.dot(bases[i], bases[i]))
    return projection

def linearCombination(num_components, weights, bases, average_loudness=0):
    lc = np.zeros(bases[0].shape[0]) #dimensionality of basis vectors
    for i in range(num_components):
        lc = lc + weights[i]*bases[i]
    lc = lc + average_loudness #add the average loudness of the segment, if not specified assume 0
    return lc
    
def recompose_spectrogram(song, plot_spectrogram=True):
    Log_power, _ = spectrogram_operations.log_power_spectrogram(song)

    Log_power = spectrogram_operations.truncate_spectrogram(Log_power) #truncate to be integer number of segments

    segmented_spectrogram = spectrogram_operations.segment_spectrogram(Log_power)

    #preprocessing, remember loudness so it can be applied after projection
    scaled_segments, loudnesses = spectrogram_operations.normalize(segmented_spectrogram)

    # calculate projections onto bases
    segmentProjections = []
    for scaled_segment in scaled_segments:
        segmentProjections.append(projection(NUM_COMPONENTS, scaled_segment, components))

    # compute linear combination of projections, adding back the loudness
    reconstructed = []
    for index, segmentProjection in enumerate(segmentProjections):
        reconstructed.append(linearCombination(
            NUM_COMPONENTS, 
            segmentProjection, 
            components, 
            loudnesses[index])
        )

    #reshape reconstruction to spectrogram dimensions
    reconstructed_spectrogram = spectrogram_operations.array_to_spectrogram_shape(np.array(reconstructed))

    if(plot_spectrogram == True):
        spectrogram_operations.plot_spectrogram(reconstructed_spectrogram, SAMPLE_RATE, HOP_SIZE, "Reconstructed Log Power Spectrogram")

    return reconstructed_spectrogram