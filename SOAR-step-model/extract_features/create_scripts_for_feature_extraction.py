import argparse
from tqdm import trange
import os
import shutil
import pandas as pd

GLOBAL_CROP_SBATCH_TEMPLATE="""#!/bin/bash
#SBATCH --partition=partition --qos=normal --account=account
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:1

#SBATCH --job-name="{model}_{num}"
#SBATCH --output=./logs/log-%j.out # STDOUT

eval "$(conda shell.bash hook)"
conda activate feature_extract

export DECORD_EOF_RETRY_MAX=204800

python3 extract_tad_feature.py \\
    --model_name {model} \\
    --model_arch {model_arch} \\
    --ckpt_path {ckpt_path} \\
    --download_root {download_root} \\
    --video_dir {video_dir} \\
    --video_lst_csv {video_lst_csv} \\
    --save_path {output_dir} \\
    --video_lst \"{video_lst}\" \\
    --fps {fps}
"""

SHELL_SCRIPT_TEMPLATE=\
"""for i in `seq 0 {batch_size} {end_num}`
do
    echo sbatch {sbatch_scripts_dir}/extract$i.sbatch
    sbatch {sbatch_scripts_dir}/extract$i.sbatch
done
"""

def main(args):
    assert args.video_dir is not None or args.video_lst_csv is not None

    args.sbatch_scripts_dir = os.path.join(args.sbatch_scripts_dir, args.model)

    if os.path.exists(args.sbatch_scripts_dir):
        shutil.rmtree(args.sbatch_scripts_dir)
    
    os.makedirs(args.sbatch_scripts_dir, exist_ok=True)

    if args.video_dir:
        video_lst = os.listdir(args.video_dir)
        video_lst = [os.path.splitext(f)[0] for f in video_lst]
    else:
        video_lst = pd.read_csv(args.video_lst_csv, dtype={'video_id': str})['video_id']

    # Remove the videos already with extracted features
    if os.path.exists(args.output_dir):
        vids_already_extracted = os.listdir(args.output_dir)
        vids_already_extracted = [v.replace('.npy', '') for v in vids_already_extracted]

        print(f'{len(vids_already_extracted)} videos already extracted')
        video_lst = set(video_lst) - set(vids_already_extracted)

    print(f'{len(video_lst)} videos to extract')
    video_lst = list(video_lst)

    sbatch_template = GLOBAL_CROP_SBATCH_TEMPLATE

    for j in trange(0, len(video_lst), args.batch_size):
        videos_subset = video_lst[j:j+args.batch_size]

        sbatch_str = sbatch_template.format(
            model=args.model,
            model_arch=args.model_arch,
            ckpt_path=args.ckpt_path,
            download_root=args.download_root,
            video_dir=args.video_dir, 
            video_lst_csv=args.video_lst_csv,
            output_dir=args.output_dir, 
            video_lst=','.join(videos_subset),
            num=j,
            fps=args.fps
        )

        sbatch_file_path = f'{args.sbatch_scripts_dir}/extract{j}.sbatch'
        with open(sbatch_file_path, 'w') as f:
            f.write(sbatch_str)

    # Create shell script
    shell_text = SHELL_SCRIPT_TEMPLATE.format(batch_size=args.batch_size, end_num=j, sbatch_scripts_dir=args.sbatch_scripts_dir)

    shell_filepath = os.path.join(args.sbatch_scripts_dir, f'queue_all.sh')
    with open(shell_filepath, 'w') as f:
        f.write(shell_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--model_arch', type=str, help='model arch to use')
    parser.add_argument('--ckpt_path', type=str, default=None, help='model ckpt path')
    parser.add_argument('--download_root', type=str, default=None, help='where model ckpts are downloaded to')
    parser.add_argument('--video_dir', type=str, default=None, help='directory of the video clips to emebd with videomae')
    parser.add_argument('--video_lst_csv', type=str, default=None, help='csv of videos to extract features for')
    parser.add_argument('--output_dir', type=str, help='where the extracted features will end up')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sbatch_scripts_dir', type=str, default='scripts/sbatch_scripts/extract')
    parser.add_argument('--local_center_crop', action='store_true')
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()

    main(args)

    
