import joblib
import numpy as np
import matplotlib.pyplot as plt

FRAME_SIZE = 32
HOP_SIZE = int(FRAME_SIZE/2)
NFB = int(FRAME_SIZE/2) + 1
FRAMES_PER_SEGMENT = 27
FLOATS_PER_SEGMENT = FRAMES_PER_SEGMENT*NFB
SAMPLE_RATE = 22050
DURATION = FRAME_SIZE * FRAMES_PER_SEGMENT / SAMPLE_RATE

model = joblib.load('fitted_pca_model.joblib')
components = model.components_

num_comps = components.shape[0]
num_rows = 3
num_cols = 4
fig, axs = plt.subplots(num_rows, num_cols)
for i in range(num_comps):
    row = i // num_cols
    col = i % num_cols

    component = np.array(components[i])
    component = component.reshape(FRAMES_PER_SEGMENT, NFB).T
    x_vals = np.linspace(0, DURATION, FRAMES_PER_SEGMENT)
    y_vals = np.linspace(0, SAMPLE_RATE/2, NFB)

    X, Y = np.meshgrid(x_vals, y_vals)
    axs[row, col].pcolor(X, Y, component, cmap='hot')
    axs[row, col].set_title(f'Principal Component {i + 1}')
    axs[row, col].set_xlabel('Time (ms)')
    axs[row, col].set_ylabel('Freq (Hz)')

plt.tight_layout()
plt.show()