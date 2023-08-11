import torch
import torchvision.utils

from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import util, openpose_utils
import itertools
import os, ntpath
import numpy as np
from collections import OrderedDict
import glob
import cv2

from SMPLDataset.human_digitalizer.bodynets import SMPL
from SMPLDataset.human_digitalizer.renders import SMPLRenderer
from SMPLDataset.human_pose2d_estimators.skeleton_visualizer import draw_skeleton

class Dance(BaseModel):
    def name(self):
        return "Pose-Guided Person Image Animation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='dance', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--netD_V', type=str, default='temporal', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for generation loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for generation loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for generation loss')
        parser.add_argument('--lambda_cx', type=float, default=0.1, help='weight for generation loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for generation loss')
        parser.add_argument('--frames_D_V', type=int, default=6, help='number of frames of D_V')

        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.add_argument('--write_ext', type=str, help='png | jpg')

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=False)

        # display
        parser.set_defaults(eval_iters_freq=1000)
        parser.set_defaults(save_latest_freq=1000)
        parser.set_defaults(save_iters_freq=1000)

        parser.set_defaults(write_ext='png')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        generator_loss = ['app_gen', 'cx_gen', 'content_gen', 'style_gen']
        GAN_loss = ['gan_Gen', 'gan_Dis']
        video_loss = ['gan_videoGen', 'gan_videoDis']
        # pose_loss = ['smpl_gen']
        pose_loss = []
        self.loss_names = generator_loss + GAN_loss + video_loss + pose_loss

        # self.visual_names = ['ref_image', 'ref_verts', 'vert_step', 'P_step', 'img_gen', 'verts_smooth', 'kpts_coord']
        self.visual_names = ['ref_image', 'ref_verts', 'vert_step', 'P_step', 'img_gen']
        self.model_names = ['G', 'D', 'D_V']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids) > 0 else torch.ByteTensor

        self.net_G = network.define_g(opt, filename='generator', image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer,
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        if len(opt.gpu_ids) > 1:
            self.net_G = torch.nn.DataParallel(self.net_G, device_ids=self.gpu_ids)

        self.flow2color = util.flow2color()
        self.convert2skeleton = openpose_utils.tensor2skeleton(spatial_draw=True)

        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        if len(opt.gpu_ids) > 1:
            self.net_D = torch.nn.DataParallel(self.net_D, device_ids=self.gpu_ids)

        self.net_D_V = network.define_d(opt, name=opt.netD_V, input_length=opt.frames_D_V, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        if len(opt.gpu_ids) > 1:
            self.net_D_V = torch.nn.DataParallel(self.net_D_V, device_ids=self.gpu_ids)

        if self.isTrain:
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss().to(opt.device)
            # self.smplloss = torch.nn.L1Loss().to(opt.device)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                filter(lambda p: p.requires_grad, self.net_D_V.parameters())),
                lr=opt.lr * opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)

            if self.opt.use_mask:
                # use mask to calculate the correctness loss for foreground content
                self.opt.lambda_correct = 2.0
        else:
            self.results_dir_base = self.opt.results_dir
        self.esp = 1e-6
        self.setup(opt)

        self.render = SMPLRenderer(image_size=opt.load_size, map_name="par").to(opt.device)
        self.smpl_model = SMPL('./SMPLDataset/checkpoints/smpl_model.pkl').eval().to(opt.device)
        self.smpl_model.requires_grad_(False)

        self.render.set_ambient_light()
        self.texture = self.render.color_textures()[None].to(opt.device)
        self.net_G.set_renderer(self.render)
        self.net_G.set_smpl_model(self.get_verts)


    def set_input(self, data):
        # move to GPU and change data types
        opt = self.opt

        bs, n_frames_total, height, width = data['gen_images'].size()
        if self.isTrain:
            self.data_name = data['gen_paths']
            self.n_frames_total = n_frames_total // opt.image_nc
            self.n_frames_load = opt.max_frames_per_gpu
            smpl_cat = data['gen_smpls'].view(bs * self.n_frames_total, -1).cuda()
            cam_result, verts_result = self.get_verts(smpl_cat)
            gen_smpl_cat = smpl_cat.view(bs, self.n_frames_total, -1)
            cam_steps = cam_result.view(bs, self.n_frames_total, -1)
            verts_steps = verts_result.view(bs, self.n_frames_total, -1, 3)

            self.ref_smpl = data['ref_smpl'].cuda()
            cam_result, verts_result = self.get_verts(self.ref_smpl)
            self.ref_cams = cam_result.view(bs, -1)
            self.ref_verts = verts_result.view(bs, -1, 3)

            gen_images = data['gen_images']
            gen_masks = data['gen_masks'] if self.opt.use_mask else None
            gen_skeletons = data['gen_skeleton']

            self.pre_cams_iter, self.pre_verts_iter = [], []
            self.fut_cams_iter, self.fut_verts_iter = [], []

            self.gen_images, self.gen_skeletons, self.gen_masks, self.gen_cams, self.gen_verts, self.gen_smpls = [], [], [], [], [], []
            for i in range(0, self.n_frames_total, self.n_frames_load):
                if i == 0:
                    self.pre_cams_iter.append(cam_steps[:, 0].view(bs, opt.cam_nc))
                    self.pre_verts_iter.append(verts_steps[:, 0, ...].view(bs, opt.vert_nc, 3))
                    self.fut_cams_iter.append(cam_steps[:, i + self.n_frames_load].view(bs, opt.cam_nc))
                    self.fut_verts_iter.append(verts_steps[:, i + self.n_frames_load, ...].view(bs, opt.vert_nc, 3))
                elif i == self.n_frames_total - self.n_frames_load:
                    self.pre_cams_iter.append(cam_steps[:, i - 1].view(bs, opt.cam_nc))
                    self.pre_verts_iter.append(verts_steps[:, i - 1, ...].view(bs, opt.vert_nc, 3))
                    self.fut_cams_iter.append(cam_steps[:, self.n_frames_total-1].view(bs, opt.cam_nc))
                    self.fut_verts_iter.append(verts_steps[:, self.n_frames_total-1, ...].view(bs, opt.vert_nc, 3))
                else:
                    self.pre_cams_iter.append(cam_steps[:, i - 1].view(bs, opt.cam_nc))
                    self.pre_verts_iter.append(verts_steps[:, i - 1, ...].view(bs, opt.vert_nc, 3))
                    self.fut_cams_iter.append(cam_steps[:, i + self.n_frames_load].view(bs, opt.cam_nc))
                    self.fut_verts_iter.append(verts_steps[:, i + self.n_frames_load, ...].view(bs, opt.vert_nc, 3))

                P_step = gen_images[:, i * opt.image_nc:(i + self.n_frames_load) * opt.image_nc]
                P_step = P_step.view(-1, self.n_frames_load, opt.image_nc, height, width)
                self.gen_images.append(P_step)

                skeleton_step = gen_skeletons[:, i * opt.structure_nc:(i + self.n_frames_load) * opt.structure_nc]
                skeleton_step = skeleton_step.view(-1, self.n_frames_load, opt.structure_nc, height, width)
                self.gen_skeletons.append(skeleton_step)

                cam_step = cam_steps[:, i:(i + self.n_frames_load)]
                cam_step = cam_step.view(bs, self.n_frames_load, opt.cam_nc)
                self.gen_cams.append(cam_step)

                vert_step = verts_steps[:, i:(i + self.n_frames_load), ...]
                vert_step = vert_step.view(bs, self.n_frames_load, opt.vert_nc, 3)
                self.gen_verts.append(vert_step)

                smpl_step = gen_smpl_cat[:, i:(i + self.n_frames_load), ...]
                smpl_step = smpl_step.view(bs, self.n_frames_load, -1)
                self.gen_smpls.append(smpl_step)
                if self.opt.use_mask:
                    mask_step = gen_masks[:, i * opt.mask_nc:(i + self.n_frames_load) * opt.mask_nc]
                    mask_step = mask_step.view(-1, self.n_frames_load, opt.mask_nc, height, width)
                    self.gen_masks.append(mask_step)

            self.ref_image = data['ref_image'].cuda()
            self.ref_skeleton = data['ref_skeleton'].cuda()
            # self.ref_cams = cam_result[-1:].view(bs, -1).contiguous()
            # self.ref_verts= verts_result[-1:].view(bs, -1, 3).contiguous()
            self.pre_gt_image = self.ref_image
            self.pre_image = None
            self.pre_skeleton = None
            self.pre_cam = None
            self.pre_vert = None
            self.pre_smpl = None
            self.pre_state = None
            self.uv_feat_pre = None
            self.pre_warp_feat_iter = None
            self.image_paths = data['gen_paths']
        else:
            self.n_frames_total = n_frames_total // opt.image_nc
            self.n_frames_load = self.n_frames_total

            self.gen_images = data['gen_images'].view(-1, self.n_frames_load, opt.image_nc, height, width).cuda()
            self.gen_skeletons = data['gen_skeleton'].view(-1, self.n_frames_load, opt.structure_nc, height, width).cuda()

            smpl_cat = data['gen_smpls'].view(bs * self.n_frames_total, -1).cuda()
            cam_result, verts_result = self.get_verts(smpl_cat)
            cam_step = cam_result.view(bs, self.n_frames_total, -1)
            self.gen_verts = verts_result.view(bs, self.n_frames_total, -1, 3)
            self.gen_cams = cam_step.view(bs, self.n_frames_load, opt.cam_nc)

            self.smpl_step = None
            self.ref_smpl = None

            self.frame_idx = data['frame_idx']

            cam_result, verts_result = self.get_verts(data['pre_post_smpl'][0].cuda())
            self.pre_cams_iter = cam_result.view(bs, -1)
            self.pre_verts_iter = verts_result.view(bs, -1, 3)

            cam_result, verts_result = self.get_verts(data['pre_post_smpl'][1].cuda())
            self.fut_cams_iter = cam_result.view(bs, -1)
            self.fut_verts_iter = verts_result.view(bs, -1, 3)

            if self.frame_idx == self.opt.start_frame + self.opt.n_frames_pre_load_test:
                self.ref_image = data['ref_image'].cuda()
                self.ref_skeleton = data['ref_skeleton'].cuda()
                smpl_cat = data['ref_smpl'].cuda()
                cam_result, verts_result = self.get_verts(smpl_cat)
                self.ref_cams = cam_result.view(bs, -1)
                self.ref_verts = verts_result.view(bs, -1, 3)
                self.pre_image = None
                self.pre_skeleton = None
                self.pre_cam = None
                self.pre_vert = None
                self.pre_smpl = None
                self.pre_state = None
                self.uv_feat_pre = None
                self.pre_warp_feat_iter = None

                name = data['ref_path'][0].split('/')[-3]
                self.opt.results_dir = os.path.join(self.results_dir_base, f'eval_{self.opt.name}', name, 'frames')
                util.mkdir(self.opt.results_dir)

            self.change_seq = data['change_seq']
            self.image_paths = data['gen_paths']
            self.ref_paths = data["ref_path"]
            # if not self.if_cross_eval(self.image_paths, self.ref_paths):
            #     self.opt.results_dir = os.path.join(self.results_dir_base,
            #                                         self.image_paths[0][0].split('/')[-2])
            # else:
            #     name = self.image_paths[0][0].split('/')[-2] + '_with_' + self.ref_paths[0].split('/')[-2]
            #     self.opt.results_dir = os.path.join(self.results_dir_base, name)

    def get_verts(self, smpl_para, get_landmarks=False):
        cam = smpl_para[:, 0:self.opt.cam_nc].contiguous()
        pose = smpl_para[:, self.opt.cam_nc:self.opt.cam_nc + self.opt.pose_nc].contiguous()
        shape = smpl_para[:, -self.opt.shape_nc:].contiguous()
        # with torch.no_grad():
        verts, kpts3d, _ = self.smpl_model(beta=shape, theta=pose, get_skin=True)
        if get_landmarks:
            X_trans = kpts3d[:, :, :2] + cam[:, None, 1:]
            kpts2d = cam[:, None, 0:1] * X_trans
            return cam, verts, kpts2d
        else:
            return cam, verts

    def if_cross_eval(self, image_paths, ref_paths):
        video_driven = image_paths[0][0].split('/')[-2]
        video_ref = ref_paths[0].split('/')[-2]
        return False if video_ref == video_driven else True

    def write2video(self, name_list):
        images = []
        for name in name_list:
            images.append(sorted(glob.glob(self.opt.results_dir + '/*_' + name + '.' + self.opt.write_ext)))

        image_array = []
        for i in range(len(images[0])):
            cat_im = None
            for image_list in images:
                im = cv2.imread(image_list[i])
                if cat_im is not None:
                    cat_im = np.concatenate((cat_im, im), axis=1)
                else:
                    cat_im = im
            image_array.append(cat_im)

        res = ''
        for name in name_list:
            res += (name + '_')
        out_name = self.opt.results_dir + '_' + res + '.mp4'
        print('write video %s' % out_name)
        height, width, layers = cat_im.shape
        size = (width, height)
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

        for i in range(len(image_array)):
            out.write(image_array[i])
        out.release()

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if name == 'P_step' or name == 'BP_step':
                    value = value[0]
                    list_value = []
                    for i in range(value.size(0)):
                        _value = value[i, -3:, ...]
                        list_value.append(_value.unsqueeze(0))
                    value = list_value
                elif '_skeleton' in name:
                    value = value[0, -3:, ...]
                elif 'flow_field' in name or 'occlusion' in name:
                    list_value = [item for sub_list in value for item in sub_list]
                    value = list_value
                elif 'ref_verts' in name:
                    verts_value = value[0][None]
                    cams_value = getattr(self, name.replace('verts', 'cams'))[0][None]
                    rd_imgs, _ = self.render.render(cams_value, verts_value, self.texture)
                    value = rd_imgs
                elif 'vert_step' in name:
                    verts_values = value[0].cuda()
                    cams_values = getattr(self, name.replace('vert', 'cam'))[0].cuda()
                    num_frames = verts_values.size(0)
                    textures = self.texture.repeat(num_frames, 1, 1, 1, 1, 1)
                    rd_imgs, _ = self.render.render(cams_values, verts_values, textures)
                    value = [item[None] for item in rd_imgs]
                elif 'verts_smooth' in name:
                    verts_values = value[0].cuda()
                    cams_values = getattr(self, name.replace('verts', 'cams'))[0].cuda()
                    num_frames = verts_values.size(0)
                    textures = self.texture.repeat(num_frames, 1, 1, 1, 1, 1)
                    rd_imgs, _ = self.render.render(cams_values, verts_values, textures)
                    value = [item[None] for item in rd_imgs]
                elif 'kpts_coord' in name:
                    # draw_skeleton(orig_img, joints, radius=6, transpose=True, threshold=0.25)
                    temp = value[0].detach().cpu().numpy()
                    temp = (temp + 1) / 2.0 * 256
                    seq, num, _ = temp.shape
                    valid = np.ones((seq, num, 1), 'float32')
                    img = np.zeros((seq, 256, 256, 3), 'uint8')
                    temp = np.concatenate((temp, valid), axis=2)
                    out = []
                    for i in range(seq):
                        out.append(draw_skeleton(orig_img=img[i], joints=temp[i], transpose=False))
                    value = out

                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        if value[i].dtype == 'uint8':
                            visual_ret[name + str(i)] = value[i]
                        else:
                            visual_ret[name + str(i)] = self.convert2im(value[i], name)
                    # visual_ret[name] = util.tensor2im(value[-1].data)
                else:
                    visual_ret[name] = self.convert2im(value, name)
        return visual_ret

    def test(self, index, print_frq):
        """Forward function used in test time"""
        self.img_gen, self.smoothed_smpl, self.pre_state, self.P_previous_recoder, \
        self.cams_smooth, self.verts_smooth, self.kpts_coord, self.kpts_map, self.uv_feat_pre, self.pre_warp_feat_iter \
            = self.net_G(self.gen_skeletons,
                         self.ref_image,
                         self.ref_skeleton,
                         self.pre_image,
                         self.pre_skeleton,
                         self.gen_cams,
                         self.gen_verts,
                         self.ref_cams,
                         self.ref_verts,
                         self.pre_cam,
                         self.pre_vert,
                         self.smpl_step,
                         self.ref_smpl,
                         self.pre_smpl,
                         self.pre_state,
                         self.pre_cams_iter, self.pre_verts_iter, self.fut_cams_iter, self.fut_verts_iter,
                         self.uv_feat_pre, self.pre_warp_feat_iter
                         )
        self.pre_image = self.img_gen[-1]
        self.pre_skeleton = self.gen_skeletons[:, -1, ...]
        self.pre_cam = self.gen_cams[:, -1, ...]
        self.pre_vert = self.gen_verts[:, -1, ...]

        self.image_paths = self.check_image_paths()

        ref_image = util.tensor2im(self.ref_image)
        ref_smpl_rgb, _ = self.render.render(self.ref_cams, self.ref_verts, self.texture)
        ref_smpl_rgb = util.tensor2im(ref_smpl_rgb)

        textures = self.texture.repeat(self.opt.n_frames_pre_load_test, 1, 1, 1, 1, 1)
        gen_smpls_rgb, _ = self.render.render(self.gen_cams[0], self.gen_verts[0], textures)
        gt = self.gen_images.squeeze(0)
        for i in range(self.opt.n_frames_pre_load_test):
            gt_img = util.tensor2im(gt[i:i+1,:,:,:])
            gen_smpl_rgb = util.tensor2im(gen_smpls_rgb[i:i+1,:,:,:])
            gen_img = util.tensor2im(self.img_gen[i])

            height, width,_ = gen_img.shape
            vis = np.zeros((height, width * 5, 3)).astype(np.uint8)
            vis[:, :width, :] = ref_image
            vis[:, width:width * 2, :] = ref_smpl_rgb
            vis[:, width * 2:width * 3, :] = gt_img
            vis[:, width * 3:width * 4, :] = gen_smpl_rgb
            vis[:, width * 4:width * 5, :] = gen_img

            outpath = os.path.join(self.opt.results_dir, ntpath.basename(self.image_paths[i]))
            util.save_image(vis, outpath)
            if index % print_frq == 0:
                print(f'Generating image {outpath}')

        # verts_value = value[0][None]
        # cams_value = getattr(self, name.replace('verts', 'cams'))[0][None]
        # rd_imgs, _ = self.render.render(cams_value, verts_value, self.texture)
        #
        # if self.frame_idx == self.opt.start_frame + self.opt.n_frames_pre_load_test:
        #     self.save_results(self.ref_image, data_name='ref', data_ext=self.opt.write_ext)
        #
        # # save the generated image
        # gen_image = torch.cat(self.img_gen, 0)
        # self.save_results(gen_image, data_name='vis', data_ext=self.opt.write_ext)
        #
        # # save the gt image
        # gt_image = self.gen_images.squeeze(0)
        # self.save_results(gt_image, data_name='gt', data_ext=self.opt.write_ext)
        #
        # # save the skeleton image
        # current_skeleton = self.convert2skeleton(self.openpose_kp[0, ...], 'COCO_17')
        # obtained_skeleton = self.convert2skeleton(self.video2d_kp[0], 'human36m_17')
        # for i in range(len(current_skeleton)):
        #     short_path = ntpath.basename(self.image_paths[i])
        #     name = os.path.splitext(short_path)[0]
        #     util.mkdir(self.opt.results_dir)
        #
        #     in_img_name = '%s_%s.%s' % (name, 'skeleton_in', self.opt.write_ext)
        #     img_path = os.path.join(self.opt.results_dir, in_img_name)
        #     skeleton_in = current_skeleton[i].astype(np.uint8)
        #     util.save_image(skeleton_in, img_path)
        #
        #     out_img_name = '%s_%s.%s' % (name, 'skeleton_out', self.opt.write_ext)
        #     img_path = os.path.join(self.opt.results_dir, out_img_name)
        #     skeleton_out = obtained_skeleton[i].astype(np.uint8)
        #     util.save_image(skeleton_out, img_path)
        #
        # if self.change_seq:
        #     name_list = ['gt', 'vis', 'skeleton_in', 'skeleton_out']
        #     self.write2video(name_list)

    def check_image_paths(self):
        names = []
        for name in self.image_paths:
            if isinstance(name, tuple):
                name = name[0]
            names.append(name)
        return names

    def update(self):
        """Run forward processing to get the inputs"""
        for i in range(len(self.gen_images)):
            self.P_step = self.gen_images[i].cuda()
            self.BP_step = self.gen_skeletons[i].cuda()
            self.cam_step = self.gen_cams[i].cuda()
            self.vert_step = self.gen_verts[i].cuda()
            self.smpl_step = self.gen_smpls[i].cuda()
            self.mask_step = self.gen_masks[i].cuda() if self.opt.use_mask else None
            self.P_gt_previous_recoder = torch.cat((self.pre_gt_image.unsqueeze(1), self.P_step[:, :-1, ...]), 1)

            pre_cams_iter = self.pre_cams_iter[i].cuda()
            pre_verts_iter = self.pre_verts_iter[i].cuda()
            fut_cams_iter = self.fut_cams_iter[i].cuda()
            fut_verts_iter = self.fut_verts_iter[i].cuda()

            self.img_gen, self.smoothed_smpl, self.pre_state, self.P_previous_recoder, \
            self.cams_smooth, self.verts_smooth, self.kpts_coord, self.kpts_map, self.uv_feat_pre, self.pre_warp_feat_iter \
                            = self.net_G(self.BP_step, 
                                        self.ref_image, 
                                        self.ref_skeleton,
                                        self.pre_image,
                                        self.pre_skeleton,
                                        self.cam_step,
                                        self.vert_step,
                                        self.ref_cams,
                                        self.ref_verts,
                                        self.pre_cam,
                                        self.pre_vert,
                                        self.smpl_step,
                                        self.ref_smpl,
                                        self.pre_smpl,
                                        self.pre_state,
                                         pre_cams_iter, pre_verts_iter, fut_cams_iter, fut_verts_iter,
                                         self.uv_feat_pre, self.pre_warp_feat_iter
                                         )

            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            self.pre_image = self.img_gen[-1].detach()
            self.pre_skeleton = self.BP_step[:, -1, ...].detach()
            self.pre_cam = self.cam_step[:, -1, ...].detach()
            self.pre_vert = self.vert_step[:, -1, ...].detach()
            self.pre_smpl = self.smpl_step[:, -1, ...].detach()
            self.pre_state = tuple([each.detach() for each in self.pre_state])
            self.pre_gt_image = self.P_step[:, -1, ...]
            self.uv_feat_pre = self.uv_feat_pre.detach()
            self.pre_warp_feat_iter = [each.detach() for each in self.pre_warp_feat_iter]

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty
        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""

        # Spatial GAN Loss
        base_function._unfreeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        real = self.P_step[:, i, ...]
        self.loss_gan_Dis = self.backward_D_basic(self.net_D, real, fake)

        # Temporal GAN Loss
        base_function._unfreeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen) - self.opt.frames_D_V + 1)
        fake = []
        real = []
        for frame in range(self.opt.frames_D_V):
            fake.append(self.img_gen[i + frame].unsqueeze(2))
            real.append(self.P_step[:, i + frame, ...].unsqueeze(2))
        fake = torch.cat(fake, dim=2)
        real = torch.cat(real, dim=2)
        self.loss_gan_videoDis = self.backward_D_basic(self.net_D_V, real, fake)

    def backward_G(self):
        """Calculate training loss for the generator"""
        loss_style_gen, loss_content_gen, loss_app_gen, loss_cx_gen = 0, 0, 0, 0

        # Calculate the Reconstruction Loss
        for i in range(len(self.img_gen)):
            gen = self.img_gen[i]
            gt = self.P_step[:, i, ...]
            loss_app_gen += self.L1loss(gen, gt)

            content_gen, style_gen, cx_gen = self.Vggloss(gen, gt)
            loss_style_gen += style_gen
            loss_content_gen += content_gen
            loss_cx_gen += cx_gen

        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec
        self.loss_cx_gen = loss_cx_gen * self.opt.lambda_cx
        # self.loss_smpl_gen = self.smplloss(self.smpl_step, self.smoothed_smpl) * 10.0

        # Spatial GAN Loss
        base_function._freeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        D_fake = self.net_D(fake)
        self.loss_gan_Gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Temporal GAN Loss
        base_function._freeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen) - self.opt.frames_D_V + 1)
        fake = []
        for frame in range(self.opt.frames_D_V):
            fake.append(self.img_gen[i + frame].unsqueeze(2))
        fake = torch.cat(fake, dim=2)
        D_fake = self.net_D_V(fake)
        self.loss_gan_videoGen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        total_loss = 0
        for name in self.loss_names:
            if name != 'gan_Dis' and name != 'gan_videoDis':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def optimize_parameters(self):
        self.update()
