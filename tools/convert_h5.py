import json
import os

import h5py
import numpy as np
import tqdm
import logging


def add_to_hdf5(group, path, update_groups=[]):
    if len(update_groups) == 0: 
        logging.info('No new group specified')
        return
    for item in tqdm.tqdm(os.listdir(path)):
        file_path = os.path.join(path, item)
        # check if file path contains elements in new_group
        if not any([g in file_path.split("/") for g in update_groups]):
            logging.info(f"Skipping {file_path}")
            print(f"Skipping {file_path}")
            continue
        print('Adding', file_path)

        if os.path.isdir(file_path):
            # Create a new group for the directory
            try:
                sub_group = group.create_group(item)
            except ValueError:
                # delete the existing group and create a new one
                del group[item]
                sub_group = group.create_group(item)
            
            add_to_hdf5(sub_group, file_path, update_groups)
        else:
            data = None
            # Create a dataset in the current group for the file
            if file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.gif'):
                # Process as binary data
                with open(file_path, 'rb') as f:
                    data = f.read()
            elif file_path.endswith('.json'):
                # Process as JSON
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                # Convert to string for storage
                json_str = json.dumps(json_data)
                data = json_str.encode('utf-8')
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    txt_data = f.read()
                data = txt_data.encode('utf-8')
            elif file_path.endswith('.csv'):
                with open(file_path, 'r') as f:
                    csv_data = f.read()
                data = csv_data.encode('utf-8')        
            elif file_path.endswith('.mp4') or file_path.endswith('.avi') or file_path.endswith('.mov'):
                with open(file_path, 'rb') as f:
                    data = f.read()
                
            if data is None:
                logging.info(f"No matched data type for {file_path}")
                continue

            try:
                group.create_dataset(item, data=np.array(data, dtype='S'))
            except ValueError:
                del group[item]
                group.create_dataset(item, data=np.array(data, dtype='S'))


if __name__ == '__main__':
    ################# Modify this section ####################
    # base path
    base_path = "/path/to/datasets/"
    # dataset path
    dataset_path = os.path.join(base_path, '<dataset_folder>')
    # hdf5 file path
    save_path = os.path.join(base_path, '<dataset_name>.h5')

    """
        [DEFAULT]: update_groups = []
        Nothing will be appended
        or
        List file/folder to be appended:
        update_groups = ["dir1", "file1"]
        or
        ALERT: every file will be rewritten!
        update_groups = [""] 
    """
    update_groups = []
    ##########################################################

    if not os.path.exists(save_path):
        hf = h5py.File(save_path, 'w')
    else:
        hf = h5py.File(save_path, 'a')
    
    add_to_hdf5(hf, dataset_path, update_groups)

    # print the groups
    logging.basicConfig(filename="convert_h5.log", level=logging.INFO)
    logging.info("Groups in the file:")
    for group in hf:
        logging.info(group)

    hf.close()