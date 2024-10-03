# Dataset Preparation

## MPII

The original dataloader is modified to support the hdf5 format. We have prepared the MPII-Cooking-2 (in hdf5 format) dataset and can be downloaded from [here](link_here).

The structure of the dataset is as follows:
```
MPII.h5/
    train.csv
    test.csv
    video/
        xxxx.mp4
        yyyy.mp4
        ...
```


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