# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av
import io
import tools.load_h5 as load_h5


def get_video_container(path_to_vid, dataset_path, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        video_data = load_h5.load_h5_file(dataset_path, path_to_vid)
        container = av.open(io.BytesIO(video_data), metadata_errors="ignore")
        if multi_thread_decode:
            container.streams.video[0].thread_type = 'AUTO'
        return container
