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
    path = '../datasets/MPII.h5'

    # check the content of the train.csv
    csv = load_h5_file(path, 'train.csv')
    print("First 5 lines of train.csv")
    print("Video Path".ljust(40), "Label")
    for clip_idx, path_label in enumerate(csv.split("\n")):
        if clip_idx < 5:
            video_path = path_label[:-3]
            label = path_label[-2:]
            print(f"{video_path:<40} {label}") 
    print("...")

    # print tree structure
    print("\nTree structure of the h5 file:")
    with h5py.File(path, 'r') as hf:
        for key in hf.keys():
            # check if the key is group
            if isinstance(hf[key], h5py.Group):
                print(f"{key}/")
                for i, subkey in enumerate(hf[key]):
                    if i < 3:
                        print(f"  {subkey}/")
                        if isinstance(hf[key], h5py.Group):
                            for j, subsubkey in enumerate(hf[key][subkey]):
                                if j < 3:
                                    print('    ', subsubkey)
                                else:
                                    print('     ...')
                                    break
                    else:
                        print('   ...')
                        break
            else:
                print(key)