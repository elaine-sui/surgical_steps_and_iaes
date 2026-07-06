import torch
import torchvision.transforms as torch_transforms

from peft import LoraConfig
from peft import get_peft_model

from . import datasets
from . import models


def build_transforms():
    rand_aug_surg = [
            [dict(type='ShearX', level=8)],
            [dict(type='ShearY', level=8)],
            [dict(type='Rotate', level=8)],
            [dict(type='TranslateX', level=8)],
            [dict(type='TranslateY', level=8)],
            [dict(type='AutoContrast', level=8)],
            [dict(type='Equalize', level=8)],
            [dict(type='Contrast', level=8)],
            [dict(type='Color', level=8)],
            [dict(type='Brightness', level=8)],
            [dict(type='Sharpness', level=8)],
    ]
    cur_transform=torch_transforms.Compose([
        torch_transforms.RandomHorizontalFlip(p=0.5),
        datasets.CustomTorchRandAugment(aug_space=rand_aug_surg),
        datasets.ColorAdjust(min_mag=0.6, max_mag=1.4),
    ])

    return cur_transform


def build_datasets(args, kwargs):
    transform = build_transforms()

    if args.use_backbone_only:
        dataset_cls = datasets.data.CocoCVSBinaryClassification
    elif not args.with_pnr:
        dataset_cls = datasets.data.CocoCVSTemporalTotalBinaryClassification
    else:
        dataset_cls = datasets.data.CocoCVSTemporalTotalBinaryClassificationWithPNR

    train_data=dataset_cls(root='/',annFile=f'{args.annotations_dir}/training.json',transform=transform, **kwargs)
    val_data=dataset_cls(root='/',annFile=f'{args.annotations_dir}/validation.json', **kwargs)
    test_data=dataset_cls(root='/',annFile=f'{args.annotations_dir}/testing.json', **kwargs)

    return train_data, val_data, test_data


def build_collator(args):

    if args.use_backbone_only:
        assert args.model in datasets.ALL_COLLATORS, f"{args.model} collator is not defined!"
        collator = datasets.ALL_COLLATORS[args.model]()
    elif not args.with_pnr:
        assert args.model in datasets.ALL_TEMPORAL_COLLATORS, f"{args.model} collator is not defined!"
        collator = datasets.ALL_TEMPORAL_COLLATORS[args.model]()
    else:
        assert args.model in datasets.ALL_TEMPORAL_COLLATORS_WITH_PNR, f"{args.model} collator is not defined!"
        collator = datasets.ALL_TEMPORAL_COLLATORS_WITH_PNR[args.model]()

    return collator


def build_dataloaders(args, kwargs):
    train_data, val_data, test_data = build_datasets(args, kwargs)

    coll= build_collator(args)

    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,collate_fn=coll)
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,collate_fn=coll)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,collate_fn=coll)   

    return train_loader, val_loader, test_loader


def build_clipwise_sliding_window_dataloader(args):
    sliding_window_dataset = datasets.data.CocoCVSTemporalTotalBinaryClassification(root='', annFile=args.ann_json)

    coll= build_collator(args)

    loader = torch.utils.data.DataLoader(sliding_window_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,collate_fn=coll)
    
    return loader


def build_framewise_sliding_window_dataloader(args):
    sliding_window_dataset = datasets.data.CocoCVSBinaryClassification(root='', annFile=args.ann_json)

    coll= build_collator(args)

    loader = torch.utils.data.DataLoader(sliding_window_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,collate_fn=coll)
    
    return loader


def build_peft_config(args):
    if args.model.startswith('DINOv2'):
            peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout= 0.05,
            r=64,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=  ["fc","head","fc1","fc2","query","key","value","proj","linear1","linear2"]
        )
    else:
        raise ValueError(f'{args.model} peft config not defined!')

    return peft_config



def build_model(args):
    assert args.model in models.ALL_MODELS, f"{args.model} is not defined!"

    print(f"Model: {args.model}")
    print(f"Use backbone only: {args.use_backbone_only}")

    if not args.use_backbone_only:
        print(f"Num TCN layers: {args.num_tcn_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Use LoRA: {not args.no_lora}")
    print(f"Use BF16: {args.use_bf16}")
    print(f"No pooling: {args.no_pooling}")
    print(f"With PNR: {args.with_pnr}")
    if args.with_pnr:
        print(f"Smoothing: {args.smoothing}")
    print(f"FPS: {args.fps}")

    model = models.ALL_MODELS[args.model](2).to("cuda")
    embed_dim = model.get_embed_dim()

    if not args.use_backbone_only and not args.with_pnr:
        model=models.TemporalTCNBinaryClassification(2,model,dim=embed_dim,num_layers=args.num_tcn_layers,dropout=args.dropout).to("cuda")
    elif not args.use_backbone_only:
        model=models.TemporalTCNBinaryClassificationWithPNR(2, model,dim=embed_dim,num_layers=args.num_tcn_layers,dropout=args.dropout, smoothing=args.smoothing).to("cuda")

    for param in model.parameters():
        param.requires_grad = True

    if not args.no_lora:
        peft_config = build_peft_config(args)

        if args.use_backbone_only:
            model = get_peft_model(model, peft_config)
        else:
            model.backbone = get_peft_model(model.backbone, peft_config)

    if args.use_bf16:
        model.backbone.to(torch.bfloat16)

    if args.checkpoint:
        print(f"Load weights from {args.checkpoint}")
        model_weights = torch.load(args.checkpoint)

        # remove PNR keys
        if 'temporal_norm.weight' in model_weights and not args.with_pnr:
            del model_weights['temporal_norm.weight']
            del model_weights['temporal_norm.bias']
            del model_weights['loc_head.weight']
            del model_weights['loc_head.bias']
        
        model.load_state_dict(model_weights)

    return model


def build_optimizer_and_scheduler(args, model):
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    return optimizer, scheduler

def build_optimizer(args, model):
    params = model.parameters()
    name = args.optimizer

    print(f"Optimizer: {name}")
    if hasattr(torch.optim, name):
        optimizer_fn = getattr(torch.optim, name)
    else:
        raise ValueError(f"torch.optim has no optimizer '{name}'.")

    optimizer = optimizer_fn(params, lr=args.lr)
    return optimizer


def build_scheduler(args, optimizer):
    name = args.scheduler

    print(f"Scheduler: {name}")
    if name is not None:
        if hasattr(torch.optim.lr_scheduler, name):
            scheduler_fn = getattr(torch.optim.lr_scheduler, name)
        else:
            raise ValueError(f"torch.optim.lr_scheduler has no optimizer '{name}'.")

        scheduler=scheduler_fn(optimizer, T_max=args.scheduler_max_iter, eta_min=1e-7)
    else:
        scheduler=None
    return scheduler


def build_image_dataloader(args, video_id):
    image_dataset = datasets.data.ImageDataset(args.frames_dir, video_id)

    coll= build_collator(args)

    loader = torch.utils.data.DataLoader(image_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=args.num_workers, collate_fn=coll)
    
    return loader