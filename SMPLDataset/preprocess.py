import sys
sys.path.insert(0, '..')
import os
import cv2
import torch
import torchvision
import subprocess
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from utils import jsonify, image_to_tensor, remove_dir, auto_unzip_fun
from human_pose2d_estimators.openpose.runner import OpenPoseRunner
from human_trackers.max_box_tracker import MaxBoxTracker
from human_pose2d_estimators.skeleton_visualizer import draw_skeleton, draw_bbox
from human_cropper import cropper
from human_pose3d_estimators.spin.runner import SPINRunner
from human_pose3d_estimators.smplify.runner import SMPLifyRunner
from human_digitalizer.bodynets import SMPL
from human_digitalizer.renders import SMPLRenderer


def process(frames, save_dir_frame, save_dir_kptsmpl, save_dir_visualization_frames, save_dir_original_frames,
            img_size, device, workers, batch_size, output_mp4_path):
    # build the tracker
    tracker = MaxBoxTracker()

    # 1 run detector and selector to find the active boxes

    pose2d_estimator = OpenPoseRunner(cfg_or_path='./SMPLDataset/human_pose2d_estimators/body25.toml' , tracker=tracker, device=device)
    pose2d_outputs = pose2d_estimator.run_over_paths(frames)
    del frames

    active_boxes = None
    crop_names = []
    crop_imgs = []
    crop_bboxes = []
    crop_kpts = []

    for kpt, path in zip(pose2d_outputs['keypoints'], pose2d_outputs['img_names']):
        if kpt['has_person']:
            active_boxes = cropper.update_active_boxes(kpt["boxes_XYXY"], active_boxes)
            fmt_active_boxes = cropper.fmt_active_boxes(active_boxes, kpt["orig_shape"], factor=1.3)
            crop_info = cropper.process_crop_img(path, fmt_active_boxes, img_size, path=True)

            crop_scale = crop_info["scale"]
            crop_start_pt = crop_info["start_pt"]
            crop_bbox = cropper.crop_resize_boxes(kpt["boxes_XYXY"], crop_scale, crop_start_pt)
            crop_kps = cropper.crop_resize_kps(kpt['keypoints'], crop_scale, crop_start_pt)

            crop_names.append(path)
            crop_imgs.append(crop_info["image"])
            crop_bboxes.append(crop_bbox)
            crop_kpts.append(crop_kps)
    del pose2d_outputs

    # 3. run smpl estimator
    SMPLify_refiner = SMPLifyRunner(cfg_or_path='./SMPLDataset/human_pose3d_estimators/smplify.toml', device=device)
    SPIN = SPINRunner(cfg_or_path='./SMPLDataset/human_pose3d_estimators/spin.toml', device=device)
    smpl_infos = SPIN.run_with_smplify(
        crop_imgs, crop_bboxes, crop_kpts, SMPLify_refiner,
        batch_size=batch_size, num_workers=workers)

    # 4. visualization & saving info
    render = SMPLRenderer(image_size=img_size).to(device)
    smpl = SMPL('./SMPLDataset/checkpoints/smpl_model.pkl').to(device)

    render.set_ambient_light()
    texs = render.color_textures().to(device)[None]

    for ids in smpl_infos['all_valid_ids'].cpu().numpy():

        smpls_results = {
            "cam": smpl_infos['all_opt_smpls'][ids, 0:3],
            "pose": smpl_infos['all_opt_smpls'][ids, 3:-10],
            "shape": smpl_infos['all_opt_smpls'][ids, -10:],
            "init_cam": smpl_infos["all_init_smpls"][ids, 0:3],
            "init_pose": smpl_infos["all_init_smpls"][ids, 3:-10],
            "init_shape": smpl_infos["all_init_smpls"][ids, -10:]
        }

        crop_name = os.path.basename(crop_names[ids])
        crop_img = crop_imgs[ids]
        crop_bbox = crop_bboxes[ids]
        crop_kpt = crop_kpts[ids]

        img_output_path = os.path.join(save_dir_frame, crop_name)
        cv2.imwrite(img_output_path, crop_img)

        crop_img_rgb = cv2.cvtColor(crop_img.copy(), cv2.COLOR_BGR2RGB)

        image_skeleton = draw_skeleton(crop_img_rgb, crop_kpt['pose_keypoints_2d'], radius=4, transpose=False, threshold=0.25)
        image_bbox = draw_bbox(crop_img_rgb, crop_bbox)

        crop_img_rgb = image_to_tensor(crop_img_rgb, device)
        image_skeleton = image_to_tensor(image_skeleton, device)
        image_bbox = image_to_tensor(image_bbox, device)

        with torch.no_grad():
            cams = smpls_results['cam'][None].float().to(device)
            pose = smpls_results['pose'][None].float().to(device)
            shape = smpls_results['shape'][None].float().to(device)
            verts, _, _ = smpl(beta=shape, theta=pose, get_skin=True)
            rd_imgs, _ = render.render(cams, verts, texs)
            sil = render.render_silhouettes(cams, verts)[:, None].contiguous()
            smpl_image = crop_img_rgb * (1 - sil) + rd_imgs * sil
        fused_images = torch.cat([crop_img_rgb, image_bbox, image_skeleton, smpl_image], dim=0)
        fused_images = torchvision.utils.make_grid(fused_images, nrow=fused_images.shape[0], normalize=True)

        visualization_path = os.path.join(save_dir_visualization_frames, crop_name)
        torchvision.utils.save_image(fused_images, visualization_path)

        out = {
            'bbox_xyxy': crop_bbox,
            'pose_keypoints_2d': crop_kpt['pose_keypoints_2d'].tolist(),
            "cam": smpls_results['cam'].tolist(),
            "pose_theta": smpls_results['pose'].tolist(),
            "shape_beta": smpls_results['shape'].tolist()
        }

        kptsmpl_output_path = os.path.join(save_dir_kptsmpl, crop_name)[:-4] + '.json'
        jsonify(kptsmpl_output_path, out)


if __name__ == "__main__":

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str('0')

    device = torch.device("cuda:0")
