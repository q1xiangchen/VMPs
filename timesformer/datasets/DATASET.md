# Dataset Preparation

## MPII-Cooking-2 Dataset
The MPII-Cooking-2 dataset and can be downloaded from [here](https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-2-dataset). After downloading the dataset, the directory structure should look like this:
```
.
├── experimentalSetup
├── videos
│   ├── s07-d72-cam-002.avi
│   └── ...
└── attributesAnnotations_MPII-Cooking-2.mat
```
We have provided the [mpii_split.py](./mpii_split.py) script to split the dataset into training, validation, and testing set. The script will generate the following directory structure:

```
MPII
├── train.csv
├── test.csv
├── train
│   ├── addV
│   │   ├── s22-d26-cam-002_8666.mp4
│   │   ├── s22-d26-cam-002_8711.mp4
│   │   └── ...
|   ├── arrangeV
│   │   ├── s28-d51-cam-002_2860.mp4
│   │   ├── s28-d51-cam-002_3142.mp4
│   │   └── ...
│   └── ...
└── test
    ├── addV
    │   ├── s08-d02-cam-002_5200.mp4
    │   ├── s08-d02-cam-002_5260.mp4
    │   └── ...
    └── ...
```

The original dataloader is modified to support the hdf5 format (in this demo we skip the validation set for simplicity). We have prepared the code in the [convert_h5.py](../../tools/convert_h5.py) script to convert the dataset to hdf5 format, where the final directory structure should look like the same as the original dataset structure but in one hdf5 file.



## Other Datasets
### To construct the dataset in hdf5 format 
1. After donwloading the dataset, prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

    ```
    path_to_video_1 label_1
    path_to_video_2 label_2
    path_to_video_3 label_3
    ...
    path_to_video_N label_N
    ```

2. Please put all annotation json files and the frame lists/videos under the same directory, then refer to the [convert_h5.py](../../tools/convert_h5.py) script to convert the dataset to hdf5 format.

3. Set `DATA.PATH_TO_DATA_DIR` to the appropriate path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.