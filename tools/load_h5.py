import h5py
import numpy as np
import io
from PIL import Image
import json


def load_h5_file(dataset_path, path):
    with h5py.File(dataset_path, 'r') as hf:
        if path.endswith('.jpg') or path.endswith('.png') or path.endswith('.gif'):
            # saved the image as raw binary, need to convert to image
            rtn = Image.open(io.BytesIO(np.array(hf[path])))
        elif path.endswith('.json'):
            # saved as a dataset string, need to convert to json dict
            rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
        elif path.endswith('.txt'):
            rtn = np.array(hf[path]).tobytes().decode('utf-8')
        elif path.endswith('.csv'):
            rtn = np.array(hf[path]).tobytes().decode('utf-8')
        elif path.endswith('.mp4'):
            rtn = np.array(hf[path])
        elif path.endswith('.avi'):
            rtn = np.array(hf[path])
        else:
            raise ValueError('Unknown file type: {}'.format(path))
        return rtn


if __name__ == "__main__":
    hf = h5py.File('./datasets/MPII.h5', 'r')
    csv = load_h5_file(hf, 'train.csv')
    for clip_idx, path_label in enumerate(csv.split("\n")):
        print(clip_idx, path_label)