import json
from xmlrpc.client import Boolean
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import torch
from typing import Any, Callable, Optional, Tuple
from transformers import AutoImageProcessor
import numpy as np
import random
import pandas as pd


class CocoCVSBinaryClassification(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        restrict_between2and5: Boolean = False,
        id2ann_csv: str = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.restrict_between2and5 = restrict_between2and5
        self.id2ann_csv = id2ann_csv

        self.annotation = json.load(open(annFile))
        image_id=[]
        image_path_dict={}
        ds_id_dict={}
        image_to_video={}
        for i in self.annotation['images']:
            image_id.append(i['id'])
            image_path_dict[i['id']]=i['img_path'] if 'img_path' in i else i['file_name']
            ds_id_dict[i['id']]=i['label'] if 'label' in i else i['ds']

            if isinstance(ds_id_dict[i['id']], bool):
                ds_id_dict[i['id']] = int(ds_id_dict[i['id']])
            image_to_video[i['id']]=i['video_id']
        self.image_path_dict=image_path_dict
        self.ds_id_dict=ds_id_dict
        self.image_to_video=image_to_video
        self.video_ids=list(set(self.image_to_video.values()))
        self.video_ids.sort()
        self.video_to_image={i:[] for i in self.video_ids}

        self.ids = list(sorted(image_id))
        for i in self.ids:
            self.video_to_image[self.image_to_video[i]].append(i)

        if self.restrict_between2and5:
            self._restrict_between2and5()

        split = annFile.split('/')[-1].split('.')[0]
        if split == 'training':
            random.seed(1234)
            random.shuffle(self.ids)

        if None not in list(ds_id_dict.values()):
            if not isinstance(list(ds_id_dict.values())[0], list):
                pos = np.array(list(ds_id_dict.values())).sum()
                print(f"Positive: {pos}")
                print(f"Negative: {len(ds_id_dict)-pos}")
            else:
                arr = np.array(list(ds_id_dict.values()))
                pos_blood = (arr[:, 0] * arr[:, 2]).sum()
                neg_blood = ((1 - arr[:, 0]) * arr[:, 2]).sum()
                pos_bile = (arr[:, 1] * arr[:, 3]).sum()
                neg_bile = ((1 - arr[:, 1]) * arr[:, 3]).sum()
                print(f"Blood Positive: {pos_blood} Negative: {neg_blood}")
                print(f"Bile Positive: {pos_bile} Negative: {neg_bile}")

    def _restrict_between2and5(self):
        df_id2ann = pd.read_csv(self.id2ann_csv).set_index('video_id')
        
        new_video_to_image = {}
        for id, image_lst in self.video_to_image.items():
            video_id = '_'.join(image_lst[0].split('_')[:-1])
            print(f"Before retrict: {len(image_lst)}")
            frame_nums = np.array(sorted([int(f.split('_')[-1]) for f in image_lst]))
            clip_csv_step = df_id2ann.loc[video_id, 'clip_csv_step']
            df_step = pd.read_csv(clip_csv_step)
            df_step['start'] = df_step.apply(lambda x: int(x['duration'].split('-')[0]), axis=1)
            df_step['end'] = df_step.apply(lambda x: int(x['duration'].split('-')[1]), axis=1)

            min_idx = None
            if '2ExposeGallbladder' in df_step['label'].values:
                min_frame = df_step[df_step['label'] == '2ExposeGallbladder']['start'].values.min().item()
                min_idx = np.where(frame_nums >= min_frame)[0][0] # get first frame in frame_nums that is greater or equal to min_frame
            max_idx = None
            if '5GallbladderDissection' in df_step['label'].values:
                max_frame = df_step[df_step['label'] == '5GallbladderDissection']['end'].values.max().item()
                max_idx = np.where(frame_nums <= max_frame)[0][-1] # get last frame in frame_nums that is less than or equal to max_frame
            
            image_lst = sorted(image_lst)
            if min_idx is not None:
                image_lst = image_lst[min_idx:]
            
            if max_idx is not None:
                image_lst = image_lst[:max_idx+1]

            print(f"After retrict: {len(image_lst)}")
            new_video_to_image[id] = image_lst
        
        self.video_to_image = new_video_to_image

    def _load_image(self, id: int) -> Image.Image:
        path = self.image_path_dict[id]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
   
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        # print(index)
        image =self._load_image(id)
        target = self.ds_id_dict[id]

        if self.transforms is not None:
            image,_= self.transforms(image,target)

        return image ,target, id

    def __len__(self) -> int:
        return len(self.ids)


class CocoCVSTemporalTotalBinaryClassification(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        restrict_between2and5: Boolean = False,
        id2ann_csv: str = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO


        self.restrict_between2and5 = restrict_between2and5
        self.id2ann_csv = id2ann_csv

        self.annotation = json.load(open(annFile))
        image_id=[]
        image_path_dict={}
        ds_id_dict={}
        image_to_video={}
        for i in self.annotation['images']:
            image_id.append(i['id'])
            image_path_dict[i['id']]=i['file_name']
            ds_id_dict[i['id']]=i['ds']
            image_to_video[i['id']]=i['video_id']
        self.image_path_dict=image_path_dict
        self.ds_id_dict=ds_id_dict
        self.image_to_video=image_to_video
        self.video_ids=list(set(self.image_to_video.values()))
        self.video_ids.sort()
        self.video_to_image={i:[] for i in self.video_ids}
        self.ids = list(sorted(image_id))
        for i in self.ids:
            self.video_to_image[self.image_to_video[i]].append(i)

        if self.restrict_between2and5:
            self._restrict_between2and5()

        targets_all = list(self.ds_id_dict.values())
        targets_all = np.array(targets_all)

        split = annFile.split('/')[-1].split('.')[0]
        print(f"Split: {split}")

        if targets_all[0] is not None:
            print(f"Positive imgs: {targets_all.sum()} \t Negative imgs: {targets_all.shape[0] - targets_all.sum()}")
        
        random.seed(1234)
        random.shuffle(self.video_ids)
        
        width, height = None, None
        for path in self.image_path_dict.values():
            random_filepath = os.path.join(self.root, path)
            if os.path.exists(random_filepath):
                img = Image.open(random_filepath)
                width, height = img.size
                break
        
        assert width is not None and height is not None
        self.black_frame = Image.new('RGB', (width, height), color=(0, 0, 0))

    def _restrict_between2and5(self):
        df_id2ann = pd.read_csv(self.id2ann_csv).set_index('video_id')
        
        new_video_to_image = {}
        for id, image_lst in self.video_to_image.items():
            video_id = '_'.join(image_lst[0].split('_')[:-1])
            print(f"Before retrict: {len(image_lst)}")
            frame_nums = np.array(sorted([int(f.split('_')[-1]) for f in image_lst]))
            clip_csv_step = df_id2ann.loc[video_id, 'clip_csv_step']
            df_step = pd.read_csv(clip_csv_step)
            df_step['start'] = df_step.apply(lambda x: int(x['duration'].split('-')[0]), axis=1)
            df_step['end'] = df_step.apply(lambda x: int(x['duration'].split('-')[1]), axis=1)

            min_idx = None
            if '2ExposeGallbladder' in df_step['label'].values:
                min_frame = df_step[df_step['label'] == '2ExposeGallbladder']['start'].values.min().item()
                min_idx = np.where(frame_nums >= min_frame)[0][0] # get first frame in frame_nums that is greater or equal to min_frame
            max_idx = None
            if '5GallbladderDissection' in df_step['label'].values:
                max_frame = df_step[df_step['label'] == '5GallbladderDissection']['end'].values.max().item()
                max_idx = np.where(frame_nums <= max_frame)[0][-1] # get last frame in frame_nums that is less than or equal to max_frame
            
            image_lst = sorted(image_lst)
            if min_idx is not None:
                image_lst = image_lst[min_idx:]
            
            if max_idx is not None:
                image_lst = image_lst[:max_idx+1]

            print(f"After retrict: {len(image_lst)}")
            new_video_to_image[id] = image_lst
        
        self.video_to_image = new_video_to_image

    def _load_image(self, id: int) -> Image.Image:
        path = os.path.join(self.root, self.image_path_dict[id])
        if not os.path.exists(path): # path doesn't exist because frame not extracted due to OOB
            # dummy black frame
            return self.black_frame
        else:
            return Image.open(path).convert("RGB")

   
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video_id = self.video_ids[index]
        image_id_list=self.video_to_image[video_id]
        image_list=[]
        for image_id in image_id_list:
            image =self._load_image(image_id)
            target = self.ds_id_dict[image_id]

            if self.transforms is not None:
                image,_= self.transforms(image,target)
            image_list.append(image)
        return image_list ,target, video_id

    def __len__(self) -> int:
        return len(self.video_ids)


class CocoCVSTemporalTotalBinaryClassificationWithPNR(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        restrict_between2and5: Boolean = False,
        id2ann_csv: str = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.restrict_between2and5 = restrict_between2and5
        self.id2ann_csv = id2ann_csv

        self.annotation = json.load(open(annFile))
        image_id=[]
        image_path_dict={}
        ds_id_dict={}
        pnr_id_dict={}
        image_to_video={}
        for i in self.annotation['images']:
            image_id.append(i['id'])
            image_path_dict[i['id']]=i['file_name']
            ds_id_dict[i['id']]=i['ds']
            image_to_video[i['id']]=i['video_id']
            pnr_id_dict[i['id']] = i['is_pnr']
        self.image_path_dict=image_path_dict
        self.ds_id_dict=ds_id_dict
        self.pnr_id_dict = pnr_id_dict
        self.image_to_video=image_to_video
        self.video_ids=list(set(self.image_to_video.values()))
        self.video_ids.sort()
        self.video_to_image={i:[] for i in self.video_ids}
        self.video_to_pnr={i:None for i in self.video_ids}
        self.ids = list(sorted(image_id))
        for i in self.ids:
            self.video_to_image[self.image_to_video[i]].append(i)
            if self.pnr_id_dict[i]:
                self.video_to_pnr[self.image_to_video[i]] = i

        if self.restrict_between2and5:
            self._restrict_between2and5()

        targets_all = list(self.ds_id_dict.values())
        targets_all = np.array(targets_all)

        split = annFile.split('/')[-1].split('.')[0]
        print(f"Split: {split}")
        print(f"Positive imgs: {targets_all.sum()} \t Negative imgs: {targets_all.shape[0] - targets_all.sum()}")
        
        random.seed(1234)
        random.shuffle(self.video_ids)

    def _restrict_between2and5(self):
        df_id2ann = pd.read_csv(self.id2ann_csv).set_index('video_id')
        
        new_video_to_image = {}
        for id, image_lst in self.video_to_image.items():
            video_id = '_'.join(image_lst[0].split('_')[:-1])
            print(f"Before retrict: {len(image_lst)}")
            frame_nums = np.array(sorted([int(f.split('_')[-1]) for f in image_lst]))
            clip_csv_step = df_id2ann.loc[video_id, 'clip_csv_step']
            df_step = pd.read_csv(clip_csv_step)
            df_step['start'] = df_step.apply(lambda x: int(x['duration'].split('-')[0]), axis=1)
            df_step['end'] = df_step.apply(lambda x: int(x['duration'].split('-')[1]), axis=1)

            min_idx = None
            if '2ExposeGallbladder' in df_step['label'].values:
                min_frame = df_step[df_step['label'] == '2ExposeGallbladder']['start'].values.min().item()
                min_idx = np.where(frame_nums >= min_frame)[0][0] # get first frame in frame_nums that is greater or equal to min_frame
            max_idx = None
            if '5GallbladderDissection' in df_step['label'].values:
                max_frame = df_step[df_step['label'] == '5GallbladderDissection']['end'].values.max().item()
                max_idx = np.where(frame_nums <= max_frame)[0][-1] # get last frame in frame_nums that is less than or equal to max_frame
            
            image_lst = sorted(image_lst)
            if min_idx is not None:
                image_lst = image_lst[min_idx:]
            
            if max_idx is not None:
                image_lst = image_lst[:max_idx+1]

            print(f"After retrict: {len(image_lst)}")
            new_video_to_image[id] = image_lst
        
        self.video_to_image = new_video_to_image

    def _load_image(self, id: int) -> Image.Image:
        path = self.image_path_dict[id]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

   
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video_id = self.video_ids[index]
        image_id_list=self.video_to_image[video_id]
        pnr_id = self.video_to_pnr[video_id]
        image_list=[]
        pnr_frame = 16
        for i, image_id in enumerate(image_id_list):
            image =self._load_image(image_id)
            target = self.ds_id_dict[image_id]

            if self.transforms is not None:
                image,_= self.transforms(image,target)
            image_list.append(image)
            if image_id == pnr_id:
                pnr_frame = i
        
        return image_list ,target, pnr_frame, video_id

    def __len__(self) -> int:
        return len(self.video_ids)


class DINOCollator:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def __call__(self, batch):
        images, labels, ids = zip(*batch)
        # print(labels)
        processed_images = self.processor(images=images, return_tensors="pt")

        if None in labels:
            labels=None
        else:
            labels = torch.tensor(labels, dtype=torch.float)

        return {"inputs":processed_images}, labels, ids


class DINObTemporalCollator:
    def __init__(self ):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def __call__(self, batch):
        images, labels, video_ids = zip(*batch)
       
        # batch size > 1
        image_tensor = []
        for images_ in images:
            image_tensor_ = self.processor(images=images_, return_tensors="pt")
            image_tensor.append(image_tensor_)
        
        if None in labels:
            labels = None
        else:
            labels = torch.tensor(labels, dtype=torch.float)

        return image_tensor, labels, video_ids


class DINObTemporalWithPNRCollator:
    def __init__(self ):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def __call__(self, batch):
        images, labels, pnr_labels, video_ids = zip(*batch)
        image_tensor = self.processor(images=images[0], return_tensors="pt") # batch size = 1

        labels = torch.tensor(labels, dtype=torch.float)
        pnr_labels = torch.tensor(pnr_labels, dtype=torch.float)

        return image_tensor, (labels, pnr_labels), video_ids