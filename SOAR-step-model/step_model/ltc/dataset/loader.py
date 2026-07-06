"""Data loader."""

from functools import partial
import torch
import pandas as pd
import numpy as np

from .utils import sequence_collate
from .video_dataset import SingleVideoDataset
from .binary_start import BinaryStart
from .generic_combined import GenericCombined

_DATASETS = {
    "generic_combined": GenericCombined,
    "binary_start": BinaryStart,
}

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    :param cfg:
    :param split:  the split of the data loader. 'train', 'test', 'eval'
    :return:
    """
    assert split in ["train", "test", "val", "all"]
    dataset_name = cfg.TRAIN.DATASET

    if split in ["train"]:
        shuffle = True
        batch_size = cfg.TRAIN.BATCH_SIZE

    elif split in ["test", "val", "all"]:
        shuffle = False
        batch_size = cfg.TRAIN.EVAL_BATCH_SIZE

    # Construct the dataset
    dataset = _DATASETS[dataset_name](cfg, split)

    custom_collate_fn = partial(sequence_collate, pad_ignore_idx=cfg.MODEL.PAD_IGNORE_IDX)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return loader, dataset

def construct_loaders_by_datasource_groups(cfg, split):
    """
    Constructs the data loader for the given dataset. by datasource
    :param cfg:
    :param split:  the split of the data loader. 'train', 'test', 'eval'
    :return:
    """
    assert split in ["train", "test", "val", "all"]
    dataset_name = cfg.TRAIN.DATASET

    if split in ["train"]:
        shuffle = True
        batch_size = cfg.TRAIN.BATCH_SIZE

    elif split in ["test", "val", "all"]:
        shuffle = False
        batch_size = cfg.TRAIN.EVAL_BATCH_SIZE

    # Construct the datasets
    path_to_master_csv = cfg.DATA.PATH_TO_MASTER_CSV
    
    df_master = pd.read_csv(path_to_master_csv)
    data_sources = list(np.unique(df_master['dataset'].values))

    data_sources = [[d] for d in data_sources]

    datasets = []
    dataloaders = []
    for source in data_sources:
        dataset = _DATASETS[dataset_name](cfg, split, source)

        custom_collate_fn = partial(sequence_collate, pad_ignore_idx=cfg.MODEL.PAD_IGNORE_IDX)
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        dataloaders.append(loader)
        datasets.append(dataset)

    return dataloaders, datasets


def construct_loader_inference(cfg, video_feats_path, active_surgery_pred_path):
    """
    Constructs the data loader for the given dataset.
    :param cfg:
    :param split:  the split of the data loader. 'train', 'test', 'eval'
    :return:
    """
    batch_size = 1
    shuffle = False

    # Construct the dataset
    dataset = SingleVideoDataset(cfg, video_feats_path, active_surgery_pred_path)

    custom_collate_fn = partial(sequence_collate, pad_ignore_idx=cfg.MODEL.PAD_IGNORE_IDX)
    
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return loader
