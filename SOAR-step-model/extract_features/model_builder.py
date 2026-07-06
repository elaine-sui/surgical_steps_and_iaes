import clip
import torch
import timm

from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_model(model_name, model_arch, download_root=None, ckpt_path=None):
    if model_name == 'CLIP':
        model, _ = clip.load(model_arch, device="cuda", download_root=download_root)
        model = model.visual
    elif model_name == 'LoVIT':
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k.replace("model.", ""):v for k,v in state_dict.items()}

        model = timm.create_model(model_arch, num_classes=0)
        err = model.load_state_dict(state_dict)
        print(err)
    elif model_name == 'DINOv2':
        model = torch.hub.load('facebookresearch/dinov2', model_arch)
    elif model_name == 'DINOv3-ViTB16':
        model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
    elif model_name == 'DINOv3-ViTL16':
        model = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
    else:
        raise NotImplementedError(f'model {model_name} {model_arch} not implemented!')
    
    return model


def to_normalized_float_tensor(vid):
    return vid.permute(0, 3, 1, 2).to(torch.float32) / 255 # (batch_size, num_channels, h, w)

class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


def _transform(n_resize, n_crop, local_center_crop=False):
    # From official CLIP repo
    if local_center_crop:
        return Compose([
        ToFloatTensorInZeroOne(),
        CenterCrop(n_crop),
        # _convert_image_to_rgb,
        # ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    return Compose([
        ToFloatTensorInZeroOne(),
        Resize(n_resize, interpolation=BICUBIC),
        CenterCrop(n_crop),
        # _convert_image_to_rgb,
        # ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



def get_transform(model_name, local_center_crop):
    if model_name in ['CLIP', 'LoVIT']:
        return _transform(224, 224, local_center_crop)
    elif model_name == 'DINOv2':
        return _transform(256, 224)
    elif model_name == 'DINOv3-ViTB16':
        return AutoImageProcessor.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
    elif model_name == 'DINOv3-ViTL16':
        return AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
    else:
        raise NotImplementedError(f'model {model_name} is not implemented!')