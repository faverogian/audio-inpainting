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
    Configure environment specs in environmentVars.ini
        - frame_size: affects how many samples are used for each STFT. Larger frame_size is better frequency resolution
            but worse time resolution
        - frames_per_segment: how many frames (stfts) compose your basis spectrotemporal surfaces. more 
            frames_per_segment means each basis function will represent a longer unit of time
        - sample_rate: sample rate used for audio processing
        - num_pca_components: how many basis functions are to be used

    # Examples
    `main.py` contains some examples of things you might want to use this for such as
    1. Performing PCA
    2. Visualizing pca basis functions
    3. Going from song directly to log power spectrogram
    4. Projection of a song onto bases and reconstruction of log power spectrogram
    5. Comparing the log power spectrogram of the original song with the reconstructed log power spectrogram (RMSE,MAE)
    6. Converting magnitude spectrogram back to audio (using phase spectrogram for timing recovery)

    # Summary of modules

    src/pca_decomposition.py:
        Handles PCA decomposition. All thats required is a function call to pca_decomposition.generate_pca_bases()
        Optional parameters are: plot_bases, which plots the basis functions on computation and save_bases_as_audio
        which saves the bases as .wav files in bases/pca_{model_identification}/component[i].wav
    
    src/pca_recomposition.py:
        Performs vector projection onto basis functions, computes the linear combination, and reconstructs the 
        magnitude spectrogram

    src/spectrogram_operations.py:
        Has functions that simplify operations relating to spectrograms:
            plot_spectrogram(spectrogram, etc) plots the passed spectrogram and configures the plot using other params
            log_power_spectrogram(song) converts a song straight to the logarithmic power spectrogram
            log_power_to_stft(spectrogram) takes a logarithmic power spectrogram and turns it back into the magnitude 
                spectrogram
            roundtrip(song) takes a song, computes the log power spectrogram, converts back to the complex spectrogram,
                then inverses the stft to convert it back into a song. This isn't particularly useful except to see 
                any effect/loss of data due to: stft quantizations, framing, windowing, and perhaps even 
                numeric (float) error since the data cannot be represented with 100% precision
            truncate_spectrogram(spectrogram) shortens the spectrogram by less than one segment to ensure the 
                spectrogram fits evenly into the fundamental time unit determined by the segment duration
            segment_spectrogram(spectrogram) splits the spectrogram into segments (time units)
            normalize(segment) takes a segment and normalizes it by the average loudness since the bases 
                were intentionally constructed to be independent of loudness. The average loudness is stored so that 
                it can later be recalled and recombined with the projections to preserve the original loudness
            array_to_spectrogram_shape(arr) takes a flattened array and reshapes it to have the same number of frequency 
                bins determined by the pca_model. The remaining dimension is equal to the number of stfts (frames) that 
                fit into the array. This is useful because the spectrograms are often flattened to 1D vectors for purposes
                such as projection and truncation, but need to be converted back to the spectrogram shape if they are to 
                be visualized
            logpowerphase_to_complexspectrogram(log_power,phase) combines log power and phase information to recover the 
                original magnitude spectrogram (stft of the original signal)
            plot_pca_bases(): (default) plots the components of the current environment configuration, but can be overloaded
                to plot any models bases with matplotlib formatting (num_rows, num_cols, components)