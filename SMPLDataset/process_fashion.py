import os
import argparse
import torch
import subprocess
import glob
from tqdm import tqdm

from utils import load_txt_file, mkdirs
import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../../Dataset/PoseTransfer/UBCFashionVideo", help="the root directory of dataset.")
    parser.add_argument("--output_dir", type=str, default="../../Dataset/UBC_fashion_smpl", help="the root directory of dataset.")
    parser.add_argument("--gpu_id", type=str, default="0", help="the gpu ids.")
    parser.add_argument("--workers", type=int, default=8, help="numbers of workers")
    parser.add_argument("--batch_size", type=int, default=64, help="numbers of batch size for SMPL")
    parser.add_argument("--image_size", type=int, default=256, help="the image size.")
    args = parser.parse_args()

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device("cuda:0")

    stages = ['train', 'test']

    train = load_txt_file(os.path.join(args.video_dir, 'fashion_train.txt'))
    test = load_txt_file(os.path.join(args.video_dir, 'fashion_test.txt'))
    train_names = [['train', os.path.basename(each)] for each in train]
    test_names = [['test', os.path.basename(each)] for each in test]
    names = train_names + test_names

    for stage, name in tqdm(names):
        output_mp4_path = os.path.join(args.output_dir, stage, name, name)
        save_dir_visualization_frames = os.path.join(args.output_dir, stage, name, 'visualization')
        mkdirs(save_dir_visualization_frames)
        save_dir_original_frames = os.path.join(args.output_dir, stage, name, 'original_frames')
        mkdirs(save_dir_original_frames)
        save_dir_frame = os.path.join(args.output_dir, stage, name, 'frames')
        mkdirs(save_dir_frame)
        save_dir_kptsmpl = os.path.join(args.output_dir, stage, name, 'kptsmpls')
        mkdirs(save_dir_kptsmpl)

        ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", "ffmpeg")

        cmd = [
            ffmpeg_exc_path,
            "-i", os.path.join(args.video_dir, stage, name),
            "-start_number", "0",
            "-hide_banner",
            "-loglevel", "error",
            "{temp_dir}/frame_%05d.png".format(temp_dir=save_dir_original_frames),
        ]

        subprocess.call(cmd)

        frames = sorted(glob.glob(os.path.join(save_dir_original_frames, '*.png')))
        preprocess.process(frames, save_dir_frame, save_dir_kptsmpl, save_dir_visualization_frames, save_dir_original_frames,
                           args.image_size, device, args.workers, args.batch_size, output_mp4_path)

