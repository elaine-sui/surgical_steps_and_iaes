import argparse
from src.builder import build_dataloaders, build_model, build_optimizer_and_scheduler, build_image_dataloader
from src import train
import os

import datetime
from dateutil import tz

def get_datetime():
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    return timestamp

def main(args):
    kwargs = {}
    model = build_model(args)

    if args.extract_features:
        for video_id in os.listdir(args.frames_dir):
            if video_id in ['sst_223', 'sst_241']: # corrupted
                continue
            data_loader = build_image_dataloader(args, video_id)
            os.makedirs(args.features_dir, exist_ok=True)
            features_path = os.path.join(args.features_dir, video_id + ".npy")
            train.extract_features(model, data_loader, features_path=features_path, use_bf16=args.use_bf16)
        exit(0)

    train_loader, val_loader, test_loader = build_dataloaders(args, kwargs)
    date = get_datetime()

    save_path = f"{args.prefix_dir}{args.output_dir}/{args.exp_name}/{date}"
    print(f"Checkpoints saved to {save_path}")

    if args.train:
        print(f"Num epochs: {args.num_epochs}")
        optimizer, scheduler = build_optimizer_and_scheduler(args, model)
        best_ckpt_path = train.train_and_evaluate(model,train_loader,val_loader,optimizer,num_epochs=args.num_epochs,save_path=save_path,scheduler=scheduler,test_loader=test_loader,use_bf16=args.use_bf16, save_pred_and_gt=args.save_pred_and_gt, fps=args.fps)
        args.checkpoint = best_ckpt_path
        model = build_model(args) # load best checkpoint
        train.evaluate(model,test_loader,save_path='/'.join(args.checkpoint.split('/')[-3:]).replace('.pt', '.pkl'), save_pred_and_gt=True, use_bf16=args.use_bf16, fps=args.fps)
    elif args.test:
        assert args.checkpoint is not None, "Need to load a checkpoint!"
        train.evaluate(model,test_loader,save_path='/'.join(args.checkpoint.split('/')[-3:]).replace('.pt', '.pkl'), save_pred_and_gt=args.save_pred_and_gt, use_bf16=args.use_bf16, fps=args.fps)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str)

    # data
    parser.add_argument('--fps', type=int, default=1)

    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--use_backbone_only', action='store_true')
    parser.add_argument('--num_tcn_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--no_pooling', action='store_true')

    parser.add_argument('--with_pnr', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--with_norm', action='store_true')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--scheduler_max_iter', type=int, default=1000*20)
    parser.add_argument('--num_epochs', type=int, default=20)

    # train and test
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_pred_and_gt', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    # extract features
    parser.add_argument('--extract_features', action='store_true')
    parser.add_argument('--features_dir', type=str, default='/path/to/soar_wellcome_save/dinov2_tcn/extracted_features')
    parser.add_argument('--frames_dir', type=str, default='/path/to/soar_wellcome_save/full_video_frames')

    parser.add_argument('--prefix_dir', type=str, default='/path/to/soar_wellcome_save/dinov2_tcn/')

    args = parser.parse_args()

    main(args)
