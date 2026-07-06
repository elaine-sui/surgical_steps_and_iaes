import os
from os.path import join
import numpy as np
import torch

from yacs.config import CfgNode

from ltc.dataset.utils import conform_temporal_sizes
from ltc.dataset.video_dataset import VideoDataset
import ltc.utils.logging as logging
import pandas as pd

logger = logging.get_logger(__name__)

class BinaryStart(VideoDataset):
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

        print(f"CLASS INDICES: {self._cfg.DATA.BACKGROUND_INDICES}")

        self._video_meta = {}
        self._path_to_data = cfg.DATA.PATH_TO_DATA_DIR
        self._override_feature_dir = cfg.DATA.OVERRIDE_FEATURE_DIR

        self._path_to_mapping_file = cfg.DATA.PATH_TO_MAPPING_FILE
        self._path_to_master_csv = cfg.DATA.PATH_TO_MASTER_CSV
        self._subfolder = cfg.DATA.SUBFOLDER
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

        if self._mode != 'all':
            df_master = df_master[df_master['mode'] == self._mode].reset_index(drop=True)

        with open(self._path_to_mapping_file, 'r') as f:
            lines = map(lambda l: l.strip().split(), f.readlines())
            action_to_idx = {action: int(str_idx) for str_idx, action in lines}

        num_videos = int(len(df_master) * self._cfg.DATA.DATA_FRACTION)
        logger.info(f"Using {self._cfg.DATA.DATA_FRACTION*100}% of {self._mode} data.")

        df_master = df_master[:num_videos].set_index('video_id')

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
        sample['features'] = torch.tensor(self._load_features(feature_path)) #, fps))  # [D, T]
        seq_length = sample['features'].shape[-1]

        targets = torch.tensor([self._cfg.DATA.BACKGROUND_INDICES[0]] * start + [self._cfg.DATA.BACKGROUND_INDICES[1]] * (end + 1 - start) + [self._cfg.DATA.BACKGROUND_INDICES[0]] * (seq_length - end - 1)).long()
        sample['targets'] = targets[::fps]
        sample['features'] = sample['features'][:, ::fps]
        # Add background to before start and after end
        # start_bg = np.array([0] * start)
        # end_bg = np.array([0] * (seq_length - end - 1)) # note: this has shape 0 since generally last step is video end
        # print(start_bg.shape[0], end_bg.shape[0])
        # targets = np.concatenate([start_bg, targets, end_bg])
        # sample['targets'] = torch.tensor(targets).long()[::fps]  # [T]
        # sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)
        sample['video_name'] = self._video_names[index]

        return sample

    def _load_features(self, feature_path: str): #, fps: int): #, start: int, end: int):
        features = np.load(feature_path)
        features = features.astype(np.float32) 
        # features = features[start:end+1] # truncate to the last annotated frame
        features = features.T # need to transpose
        # features = features[:, ::fps]  # [D, T]
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
                phase_id = action_to_idx[label]

                items = {'Frame': list(range(start, end)), 'phase_id': [phase_id] * (end - start)}
                dfs.append(pd.DataFrame(items))
            
            df = pd.concat(dfs)
            df = df.sort_values(by='Frame')
        else:
            # print(segm_path)
            df = pd.read_csv(segm_path, sep='\t')

            key = 'Errors' if self._cfg.MODEL.ERROR_DETECTION is not None else 'Phase'

            try:
                df['phase_id'] = df.apply(lambda x: action_to_idx[x[key]], axis=1)
            except:
                import pdb; pdb.set_trace()
                print("error goes beyond step boundary!")

        start = df.iloc[0]['Frame']
        end = df.iloc[-1]['Frame']

        return df['phase_id'].values, start, end



    
    


