# A Repository for Audio Analysis and PCA Feature Extraction

## Pipeline:

### I. Obtain spectrogram
    Song to samples
    Combine samples into frame
    Apply window on frame to remove high frequency artifacts
    Compute fft on frame (stft)
    Hop to next frame, which overlaps with current frame

### II. Compute power spectrogram and log power spectrogram
    PowerSpectrum = STFT^2 (power is proportional to square of amplitude)
    LogPowerSpectrum = log(PowerSpectrum) (audio perception is logarithmic)

### III. Partition spectrogram into segments of fixed length
    Determined by frame size and number of frames desired per segment
    Each segment will have frame_size/2 + 1 frequency bins and SegmentSize (measured in number of frames) time bins
    The total number of data entries per segment is then NumberOfFrequencyBins * SegmentSize
    The segment represents a duration equal to FrameSize * SegmentSize / SampleRate seconds

### IV. Reshape partitioned spectrograms into vectors for PCA processing
    The spectrotemporal surface is 3 dimensional, a 2D matrix where the entry is the surface height
    For PCA it is convenient if this is flattened into a 1D vector where the entry is the amplitude

### V. Perform PCA on the reshaped vectors to compute the n most important basis functions
    Find the vectors that maximize variance amongst the dataset as these will be most representative of the "direction" songs commonly go
    These will be the most descriptive features

### VI. Convert the vectors back into spectrotemporal surfaces and plot them
    The resulting basis functions/components are one dimensional projections on the original basis vectors (frequency-time bins)
    Reshape the vector back to the shape of the segments to develop a sonic intuition of the features

## Usage
    Place training data in resources
    Currently the main file is configured to only read .wav files since they are generally uncompressed and better quality
    This is easily configurable in the main file
    main.py calculates spectrograms, segments them, performs pca to find basis spectrogram segments, and saves it
    loadPca.py loads and plots a saved model
    spectrogram.py plots a spectrogram for an audio file