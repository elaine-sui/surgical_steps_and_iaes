import os
import numpy as np
import torch

from yacs.config import CfgNode

from ltc.dataset.utils import conform_temporal_sizes
from ltc.dataset.video_dataset import VideoDataset
import ltc.utils.logging as logging
import pandas as pd
import random
import math

logger = logging.get_logger(__name__)

class GenericCombined(VideoDataset):
    def __init__(self, cfg: CfgNode, mode: str, source: list = None):
        assert mode in [
            "train",
            "test",
            "val",
            "all",
        ], "Split '{}' not supported".format(mode)

        if mode == 'train':
            mode = 'training'
        elif mode == 'val':
            mode = 'validation'
        elif mode == 'test':
            mode = 'testing'

        self._mode = mode
        self._datasource = source
        self._cfg = cfg

        self._video_meta = {}
        self._path_to_data = cfg.DATA.PATH_TO_DATA_DIR
        self._override_feature_dir = cfg.DATA.OVERRIDE_FEATURE_DIR
        self._override_feature_fps = cfg.DATA.OVERRIDE_FEATURE_FPS
        print(f"Override feature fps: {self._override_feature_fps}")

        if self._override_feature_dir is not None:
            print(f"Override feature dir: {self._override_feature_dir}")
        
        self._path_to_mapping_file = cfg.DATA.PATH_TO_MAPPING_FILE
        self._path_to_master_csv = cfg.DATA.PATH_TO_MASTER_CSV
        self._subfolder = cfg.DATA.SUBFOLDER
        
        self._leave_one_out_test_split = cfg.DATA.LEAVE_ONE_OUT_TEST_SPLIT
        self._cross_val_test_split = cfg.DATA.CROSS_VAL_TEST_SPLIT
        self._num_cross_val_splits = cfg.DATA.NUM_CROSS_VAL_SPLITS
        self._cross_val_validation_size = cfg.DATA.CROSS_VAL_VALIDATION_SIZE
        
        if self._leave_one_out_test_split is not None:
            self._leave_one_out_test_split = str(self._leave_one_out_test_split)
            indices = list(pd.read_csv(self._path_to_master_csv)['video_id'].values)
            test_idx = indices.index(self._leave_one_out_test_split)
            val_idx = len(indices) - 1 if test_idx == 0 else test_idx - 1
            self._val_id = indices[val_idx]

            print(f"Leave one out test: {self._leave_one_out_test_split}")
            print(f"Leave one out val: {self._val_id}")
        elif self._cross_val_test_split is not None:
            print(f"Running cross-validation! Test split {self._cross_val_test_split}")
            indices = list(pd.read_csv(self._path_to_master_csv)['video_id'].values)
            random.seed(1234)
            random.shuffle(indices)

            split_size = math.ceil(len(indices) / self._num_cross_val_splits)

            self._test_indices = indices[split_size * (self._cross_val_test_split - 1): split_size * self._cross_val_test_split]
            train_indices = indices[:split_size * (self._cross_val_test_split - 1)] + indices[split_size * self._cross_val_test_split:]

            self._train_indices = train_indices[:-self._cross_val_validation_size]
            self._val_indices = train_indices[-self._cross_val_validation_size:]

            print("Val")
            print(self._val_indices)

            print("Test")
            print(self._test_indices)

        self._construct_loader()
        self._dataset_size = len(self._path_to_features)

    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        master_csv = self._path_to_master_csv

        assert os.path.isfile(master_csv), f"Master CSV {master_csv} not found."

        df_master = pd.read_csv(master_csv, dtype={'video_id': str})

        if self._datasource is not None:
            print(f"Restrict to datasource {self._datasource}")
            df_master = df_master[df_master['dataset'].isin(self._datasource)]

        if self._leave_one_out_test_split is not None:
            if self._mode == 'testing':
                df_master = df_master[df_master['video_id'] == self._leave_one_out_test_split].reset_index(drop=True)
            elif self._mode == 'validation':
                df_master = df_master[df_master['video_id'] == self._val_id].reset_index(drop=True)
            else:
                df_master = df_master[~df_master['video_id'].isin([self._val_id, self._leave_one_out_test_split])].reset_index(drop=True)
        elif self._cross_val_test_split is not None: # cross-validation!
            if self._mode == 'testing':
                df_master = df_master[df_master['video_id'].isin(self._test_indices)].reset_index(drop=True)
            elif self._mode == 'validation':
                df_master = df_master[df_master['video_id'].isin(self._val_indices)].reset_index(drop=True)
            else:
                df_master = df_master[df_master['video_id'].isin(self._train_indices)].reset_index(drop=True)
        else:
            if self._mode != 'all':
                df_master = df_master[df_master['mode'] == self._mode].reset_index(drop=True)

        with open(self._path_to_mapping_file, 'r') as f:
            lines = map(lambda l: l.strip().split(), f.readlines())
            action_to_idx = {action: int(str_idx) for str_idx, action in lines}
        
        print(action_to_idx)

        num_videos = int(len(df_master) * self._cfg.DATA.DATA_FRACTION)
        logger.info(f"Using {self._cfg.DATA.DATA_FRACTION*100}% of {self._mode} data.")

        df_master = df_master[:num_videos].set_index('video_id')

        print(f"Dataset len: {len(df_master)}")

        self._path_to_features = []
        self._segmentations = []
        self._video_names = []
        self._start_and_ends = []
        self._fps = []

        for video_id, row in df_master.iterrows():

            if self._override_feature_dir is not None:
                feat_path = os.path.join(self._override_feature_dir, video_id + ".npy")
            else:
                feat_path = row['feat_path']
            assert os.path.isfile(feat_path), f"Feature {feat_path} not found."

            self._path_to_features.append(feat_path)
            gt_path = row['gt_path']
            segs, start, end = self._load_segmentations(gt_path, action_to_idx)
            self._segmentations.append(segs)
            self._start_and_ends.append((start, end))
            self._video_names.append(video_id)
            self._fps.append(row['fps'])
    
    def __getitem__(self, index: int):
        """

        :param index:
        :return: sample dict containing:
         'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
         'targets': torch.Tensor [batch_size, sequence_length]
        """
        sample = {}
        feature_path = self._path_to_features[index]
        start, end = self._start_and_ends[index]
        fps = self._fps[index]
        sample['features'] = torch.tensor(self._load_features(feature_path, fps, start, end))  # [D, T]
        seq_length = sample['features'].shape[-1]

        targets = self._segmentations[index]
        sample['targets'] = torch.tensor(targets).long()[::fps]  # [T]

        # print(sample['targets'].shape, sample['features'].shape, feature_path)

        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)
        sample['video_name'] = self._video_names[index]

        return sample

    def _load_features(self, feature_path: str, fps: int, start: int, end: int):
        features = np.load(feature_path)
        features = features.astype(np.float32) 

        if self._override_feature_fps is not None:
            start = int(start / fps)
            end = int(end / fps)
            fps = self._override_feature_fps

        features = features[start:end+1] # truncate to the last annotated frame
        features = features.T # need to transpose
        features = features[:, ::fps]  # [D, T]

        # normalize features for training stability?
        if self._cfg.DATA.NORMALIZE_FEATS:
            features = features / np.linalg.norm(features, axis=0, keepdims=True)

        return features

    def _load_segmentations(self, segm_path, action_to_idx):
        if segm_path.split('.')[1] == 'csv':
            df_seg = pd.read_csv(segm_path)
            # Convert step_csv to framewise df
            
            dfs = []
            for i, row in df_seg.iterrows():
                start, end = row['duration'].split('-')
                start, end = int(start), int(end)

                label = row['label'].replace('.', '').replace(' ', '')
                phase_id = action_to_idx[label] if label in action_to_idx else None

                if phase_id is not None:
                    items = {'Frame': list(range(start, end)), 'phase_id': [phase_id] * (end - start)}
                    dfs.append(pd.DataFrame(items))
                else:
                    print(f"Skip {label} annotations")
            
            if len(dfs) == 0:
                print(segm_path)
                import pdb; pdb.set_trace()
                print()

            df = pd.concat(dfs)
            df = df.sort_values(by='Frame')
        else:
            # print(segm_path)
            df = pd.read_csv(segm_path, sep='\t')

            key = 'Errors' if self._cfg.MODEL.ERROR_DETECTION is not None else 'Phase'

            try:
                df['phase_id'] = df.apply(lambda x: action_to_idx[x[key]] if x[key] in action_to_idx else None, axis=1)
            except:
                import pdb; pdb.set_trace()
                print("error goes beyond step boundary!")

        # Remove all the rows with no phase_id
        df = df.dropna(axis=0)

        start = df.iloc[0]['Frame']
        end = df.iloc[-1]['Frame']

        if len(df) != end - start + 1:
            print(segm_path)
            import pdb; pdb.set_trace()
            print()
        assert len(df) == end - start + 1, segm_path

        return df['phase_id'].values, start, end



    
    


