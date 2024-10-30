import h5py 
import threading
import queue
import os
import cv2
from tqdm import tqdm

def read(var):
    obj = file[var]
    return obj[:].tobytes().decode('utf-16')

def create_video_folder(filename, activity, start_frame, end_frame, path):
    # read the raw video and save the frames
    video_path = os.path.join(raw_video_path, filename + ".avi")

    # create the folder for the video
    folder = os.path.join(path, activity)
    os.makedirs(folder, exist_ok=True)
    filename += f"_{start_frame}"
    folder = os.path.join(folder, filename) + ".mp4"
    
    # save the video clip from the start frame to the end frame as a video to path
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(folder, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frame_number = start_frame
    while frame_number <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_number += 1
    cap.release()
    out.release()


if __name__ == "__main__":
    # ----------------- Modify this section ----------------- #
    annotation_path = 'attributesAnnotations_MPII-Cooking-2.mat'
    split_path = 'experimentalSetup'
    raw_video_path = 'videos'
    n_threads = 16
    # ----------------- Modify this section ----------------- #

    # Get number of activities
    activity = set()
    attribute = set()
    print("Getting the activity set...")
    with h5py.File(annotation_path, 'r') as file:
        
        activities = file['/annos/activity']
        attributes = file['/annos/attributeMap']

        for i in range(activities.shape[0]):
            activity.add(read(activities[i][0]))

        for i in range(attributes.shape[0]):
            attribute.add(read(attributes[i][0]))

    # annos.attributeMap(ismember(annos.attributeMap, unique(annos.activity)))
    # refer to: https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-2-dataset#:~:text=The%20file%20%E2%80%98attributesAnnotations_MPII,unique(annos.activity)))
    for acti in activity.copy():
        if acti not in attribute:
            activity.remove(acti)

    print(f"number of activities in videos: {len(activity)}\n")

    # Get the annotations
    print("Getting the annotations...")
    dic = {}
    with h5py.File(annotation_path, 'r') as file:
        filenames = file['/annos/fileName']
        start_frames = file['/annos/startFrame']
        end_frames = file['/annos/endFrame']
        activities = file['/annos/activity']

        for i in range(filenames.shape[0]):
            temp_activity = read(activities[i][0])
            if temp_activity not in activity: continue
            
            filename = read(filenames[i][0])
            start_frame = int(start_frames[i][0])
            end_frame = int(end_frames[i][0])

            if dic.get(filename) == None:
                dic[filename] = [(temp_activity, start_frame, end_frame)]
            else:
                dic[filename].append((temp_activity, start_frame, end_frame))
    print(f"number of videos: {len(dic)}\n")

    # create the dataset folder
    dataset_path = 'MPII'
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print("Creating the csv files...")
    # get all training filenames
    train_filenames, val_filenames, test_filenames = [], [], []
    for split_file, filenames_list in zip(['sequencesTrainAttr.txt', 'sequencesVal.txt', 'sequencesTest.txt'], [train_filenames, val_filenames, test_filenames]):
        with open(os.path.join(split_path, split_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                filename = line.split()[0] + "-cam-002"
                filenames_list.append(filename)
            
    print(f"number of training videos: {len(train_filenames)}")
    print(f"number of validation videos: {len(val_filenames)}")
    print(f"number of test videos: {len(test_filenames)}\n")

    # create the csv files
    print("Creating the csv files...\n")
    ordered_activities = sorted(list(activity))
    for type, filenames_list in zip(['train', 'val', 'test'], [train_filenames, val_filenames, test_filenames]):
        with open(os.path.join(dataset_path, f'{type}.csv'), 'w') as f:
            for filename in filenames_list:
                if dic.get(filename) == None: Warning(f"{filename} not found in the dataset")
                for tuple in dic[filename]:
                    activity, start_frame, end_frame = tuple
                    activity_index = ordered_activities.index(activity)
                    save_string = f"{type}/{activity}/{filename}_{start_frame}.mp4 {activity_index}"
                    f.write(f"{save_string}\n")
            f.seek(f.tell() - 1)
            f.truncate()

    print("Creating the video folders...")
    # create the video folders
    for type in ['Test', 'Train', 'Val']:
        if type == 'Train':
            filenames = train_filenames
            path = train_path
        elif type == 'Val':
            filenames = val_filenames
            path = val_path
        else:
            filenames = test_filenames
            path = test_path

        # use multi-threading to create the video folders in parallel
        total_task = sum(len(dic[filename]) for filename in filenames if dic.get(filename) != None)
        with tqdm(total=total_task, desc=f"Creating {type} video folders") as pbar:
            def worker():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    filename, activity, start_frame, end_frame = item
                    create_video_folder(filename, activity, start_frame, end_frame, path)
                    pbar.update(1)
                    q.task_done()

            q = queue.Queue()
            threads = []
            for i in range(n_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
                
            for filename in filenames:
                if dic.get(filename) == None: Warning(f"{filename} not found in the dataset")
                for tuples in dic[filename]:
                    activity, start_frame, end_frame = tuples
                    if os.path.exists(f"{path}/{activity}/{filename}_{start_frame}.mp4"):
                        continue
                    q.put((filename, activity, start_frame, end_frame))

            # block until all tasks are done
            q.join()

            # stop workers
            for i in range(n_threads):
                q.put(None)

            for t in threads:
                t.join()

    print("Done!")