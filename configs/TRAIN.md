# 🔑 Baseline TimeSformer

Please refer to the [README.md](../README.md) for installation instructions, dataset preparation and model checkpoints.

## Training the Default TimeSformer

Training the default TimeSformer that uses divided space-time attention, and operates on 8-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/MPII/baseline.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 4 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply add

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

To the yaml configs file, then you do not need to pass it to the command line every time.

> * For all experiments, we recommend editing the configuration files in [`configs/`](./) to suit your needs. 
> * We also provide the option to use wandb for logging. To enable wandb, add `WANDB.ENABLE True` to the configuration file.

## Using a Different Number of GPUs

If you want to use a smaller number of GPUs, you need to modify .yaml configuration files in [`configs/`](./). Specifically, you need to modify the `NUM_GPUS`, `TRAIN.BATCH_SIZE`, `TEST.BATCH_SIZE`, `DATA_LOADER.NUM_WORKERS` entries in each configuration file. The BATCH_SIZE entry should be the same or higher as the NUM_GPUS entry. In [`configs/MPII/baseline.yaml`](./MPII/baseline.yaml), we provide a sample configuration file for a 2 GPU setup.

## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via `TEST.CHECKPOINT_FILE_PATH`.
```
python tools/run_net.py \
  --cfg configs/MPII/baseline_test.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

## Finetuning

To finetune from an existing PyTorch checkpoint add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
TRAIN.FINETUNE True
```

# 🌺 VMPs + TimeSformer

## Finetuning the TimeSformer with VMPs

To finetune the TimeSformer with VMPs, you can use the following command:

```
python tools/run_net.py \
  --cfg configs/MPII/VMPs.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TRAIN.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
``` 
**NOTE:** The `DATA.NUM_FRAMES` entry in the configuration file should be **increased by 1** for VMPs layer. Also, set the appropriate `VMPS.EXP_NAME` and `VMPS.PENALTY_WEIGHT` for hyperparameter tuning.

## Inference with VMPs

To test the TimeSformer with VMPs, you can use the following command:

``` 
python tools/run_net.py \
  --cfg configs/MPII/VMPs_test.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
```

# 🔅 Environment

The code was developed using python 3.7 on Ubuntu 20.04. For training/testing, we used 2 GPUs on a single Tesla V100 GPU compute node. Other platforms or GPU cards have not been fully tested.
