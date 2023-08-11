import os
import argparse
from pytorch_fid.fid import FID
from pytorch_lpips import lpips
import time
from skimage.io import imread
from skimage.measure import compare_ssim
import numpy as np
import torch
from tqdm import tqdm
import git
import math
from PIL import Image
import sys; sys.path.extend(['.', 'evaluation'])
from evaluation.pytorch_fvd.calc_metrics_for_dataset import calc_metrics_for_dataset as fvd

class Evaluation:

    def __init__(self, args):
        self.args = args
        self.fid_model = FID(bs=64, cuda=True)
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        self.need2split = True

        self.paths = []
        for path, subdirs, files in os.walk(self.args.gen_dir):
            for name in files:
                if name.endswith('.png'):
                    self.paths.append(os.path.join(path, name))
        self.paths.sort()
        print(f'Total images {len(self.paths)}')

        self.score_ssim = None
        self.score_fid = None
        self.score_lpips = None
        self.score_l1 = None
        self.score_psnr = None
        self.score_fvd = None

    def compute(self, height=256, width=256, gt_index=3, gen_index=5):
        if 'fid' in self.args.metrics:
            print("Computing FID score...")
            self.compute_fid()
            print("FID score %s" % self.score_fid)

        lpips = []
        ssim = []
        l1 = []
        psnr = []
        for i in tqdm(range(len(self.paths))):
            img = imread(self.paths[i]).astype(np.float32)
            gt_imgs = np.array((img[:height, (gt_index - 1) * width:(gt_index) * width, :]))
            gen_imgs = np.array((img[:height, (gen_index - 1) * width:, :]))

            if 'lpips' in self.args.metrics:
                lpips.append(self.compute_lpips(gen_imgs, gt_imgs))
            if 'ssim' in self.args.metrics:
                ssim.append(self.compute_ssim(gen_imgs, gt_imgs))
            if 'l1' in self.args.metrics:
                l1.append(self.compute_l1(gen_imgs, gt_imgs))
            if 'psnr' in self.args.metrics:
                psnr.append(self.compute_psnr(gen_imgs, gt_imgs))
            if 'fvd' in self.args.metrics:
                dirs = self.paths[i].split(os.sep)
                frame_path = os.path.join(self.args.fake_frame_dir, dirs[-3], dirs[-1])
                img = Image.fromarray(gen_imgs.astype('uint8'), 'RGB')
                img.save(frame_path)

        if 'lpips' in self.args.metrics:
            self.score_lpips = np.mean(lpips)
            print("LPIPS score %s" % self.score_lpips)

        if 'ssim' in self.args.metrics:
            self.score_ssim = np.mean(ssim)
            print("SSIM score %s" % self.score_ssim)

        if 'l1' in self.args.metrics:
            self.score_l1 = np.mean(l1)
            print("L1 score %s" % self.score_l1)

        if 'psnr' in self.args.metrics:
            self.score_psnr = np.mean(psnr)
            print("PSNR score %s" % self.score_psnr)

        if 'fvd' in self.args.metrics:
            print("Computing FVD score...")
            results = []
            metric_run_pairs = {'fvd2048_16f': 100, 'fvd2048_128f': 50, 'fvd2048_128f_subsample8f': 50}
            source_dir = [self.args.training_frame_dir, self.args.testing_frame_dir]
            with tqdm(total=len(metric_run_pairs)*len(source_dir)) as pbar:
                for source in source_dir:
                    for m, n in metric_run_pairs.items():
                        res = fvd(self.args, [m], source, self.args.fake_frame_dir, mirror=0,
                                   resolution=256, gpus=1, verbose=0, use_cache=False, num_runs=n)
                        sub_res = {
                            'name': f"{source.split('/')[-1].replace('_frames', '')}_{m}_{n}",
                            f'{m}_mean': res['results'][f'{m}_mean'],
                            f'{m}_std': res['results'][f'{m}_std'],
                        }
                        results.append(sub_res)
                        pbar.update(1)
            self.score_fvd = results
            print("FVD scores: ")
            for each in results:
                print(each)

    def summary(self, verbose=True):
        out = {}
        for each in self.__dict__:
            if 'score' in each:
                out[each] = getattr(self, each)
        if verbose:
            print(out)
        return out

    def save(self, content):
        result_log = os.path.join(self.args.gen_dir, 'result_log.txt')
        print(f'Evaluation result is saved to {result_log}')
        with open(result_log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Evaluation Result (%s) ================\n' % now)
            log_file.write('%s\n' % content)

    def compute_fid(self, height=256, width=256, index=5):
        train_dir = os.path.join(self.args.gt_root, 'train')
        npz = os.path.join(self.args.gt_root, 'fid_stat.npz')
        self.score_fid = self.fid_model.calculate_from_disk(self.paths, npz, train_dir,
                                                            need2split=self.need2split, height=height, width=width, index=index)

    def compute_lpips(self, gen_imgs, gt_imgs):
        gt_imgs = lpips.im2tensor(gt_imgs).cuda()
        gen_imgs = lpips.im2tensor(gen_imgs).cuda()
        with torch.no_grad():
            dist01 = self.lpips_model.forward(gen_imgs, gt_imgs)
            return dist01.cpu().data.numpy()

    def compute_ssim(self, gen_imgs, gt_imgs):
        return compare_ssim(gt_imgs, gen_imgs, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=gen_imgs.max() - gen_imgs.min())

    def compute_l1(self, gen_imgs, gt_imgs):
        return np.abs(2 * (gt_imgs / 255.0 - 0.5) - 2 * (gen_imgs / 255.0 - 0.5)).mean()

    def compute_psnr(self, gen_imgs, gt_imgs):
        mse = np.mean((gt_imgs - gen_imgs) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--gt_root', help='Path to ground truth data', default='./dataset/iPER_smpl', type=str)
    parser.add_argument('--gen_dir', help='Path to output data', default=None, type=str)
    parser.add_argument('--bs', help='batch size', default=64, type=int)
    parser.add_argument('--name', help='name of the experiment', default='dancefashion', type=str)
    parser.add_argument('--metrics', help='name of metrics', default='fid,ssim,lpips,l1,psnr,fvd', type=str)
    args = parser.parse_args()

    args.metrics = args.metrics.split(',')

    if args.gen_dir is None:
        # append git tag in project name
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        args.gen_dir = './eval_results/' + f'eval_{args.name}'

    temp = list(os.path.split(args.gen_dir))
    temp[-1] = f'fake_{temp[-1]}'
    args.fake_frame_dir = os.path.join(*temp)
    videos = os.listdir(args.gen_dir)
    for v in videos:
        if v.endswith('.mp4'):
            os.makedirs(os.path.join(args.fake_frame_dir, v), exist_ok=True)

    args.training_frame_dir = os.path.join(args.gt_root, 'train_frames')
    if not os.path.exists(args.training_frame_dir):
        videos = os.listdir(os.path.join(args.gt_root, 'train'))
        for v in videos:
            if v.endswith('.mp4'):
                source_path = os.path.join(args.gt_root, 'train', v, 'frames')
                target_path = os.path.join(args.training_frame_dir, v)
                os.makedirs(target_path, exist_ok=True)
                os.system(f'cp -rf {source_path}/* {target_path}')

    args.testing_frame_dir = os.path.join(args.gt_root, 'test_frames')
    if not os.path.exists(args.testing_frame_dir):
        videos = os.listdir(os.path.join(args.gt_root, 'test'))
        for v in videos:
            if v.endswith('.mp4'):
                source_path = os.path.join(args.gt_root, 'test', v, 'frames')
                target_path = os.path.join(args.testing_frame_dir, v)
                os.makedirs(target_path, exist_ok=True)
                os.system(f'cp -rf {source_path}/* {target_path}')

    metric = Evaluation(args)
    metric.compute()
    content = metric.summary(True)
    metric.save(content)