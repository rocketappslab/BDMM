from util.util import morph
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *
from model.networks.ModulatedDCN import ModulatedDCN2dBLKv2, ToRGB, StyledConv

######################################################################################################
# Pose-Guided Person Image Animation
######################################################################################################
class DanceGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):
        super(DanceGenerator, self).__init__()

        self.x_ref = PoseSourceNet(image_nc, ngf, img_f, layers+3, norm, activation, use_spect, use_coord)
        self.x_encoder = PoseSourceNet(image_nc + structure_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)
        self.warp_encoder = PoseSourceNet(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)
        self.x_rcnn1 = ResBlock(256*2, output_nc=256, norm_layer=norm, nonlinearity=activation,learnable_shortcut=False)
        self.x_rcnn2 = ResBlock(256, output_nc=256, norm_layer=norm, nonlinearity=activation,learnable_shortcut=True)

        self.for_rcnn64 = ResBlock(128*2, output_nc=128, norm_layer=norm, nonlinearity=activation, learnable_shortcut=False)
        self.back_rcnn64 = ResBlock(128*2, output_nc=128, norm_layer=norm, nonlinearity=activation, learnable_shortcut=False)

        self.for_rcnn128 = ResBlock(64*2, output_nc=64, norm_layer=norm, nonlinearity=activation, learnable_shortcut=False)
        self.back_rcnn128 = ResBlock(64*2, output_nc=64, norm_layer=norm, nonlinearity=activation, learnable_shortcut=False)

        self.conv1up = StyledConv(256, 128, 3, 512, upsample=True)
        self.directconv1 = StyledConv(128, 128, 3, 512, upsample=False, demodulate=False)
        self.forconv1 = ModulatedDCN2dBLKv2(in_channel=128, out_channel=128, style_dim=512, flow_in_channel=256, flow_out_channel=128)
        self.backconv1 = ModulatedDCN2dBLKv2(in_channel=128, out_channel=128, style_dim=512, flow_in_channel=256, flow_out_channel=128)

        self.conv2up = StyledConv(128, 64, 3, 512, upsample=True)
        self.directconv2 = StyledConv(64, 64, 3, 512, upsample=False, demodulate=False)
        self.forconv2 = ModulatedDCN2dBLKv2(in_channel=64, out_channel=64, style_dim=512, flow_in_channel=128, flow_out_channel=64)
        self.backconv2 = ModulatedDCN2dBLKv2(in_channel=64, out_channel=64, style_dim=512, flow_in_channel=128, flow_out_channel=64)

        self.conv3up = StyledConv(64, 64, 3, 512, upsample=True)
        self.directconv3 = StyledConv(64, 64, 3, 512, upsample=False, demodulate=False)

        self.decoder256 = ToRGB(64, style_dim=512, upsample=False)
        self.out = nn.Tanh()

    def forward(self, BP_frame_step, P_ref, BP_ref, P_pre, BP_pre,
                cam_frame_step, vert_frame_step,
                cam_ref, vert_ref,
                cam_pre, vert_pre,
                smpl_step, ref_smpl,
                pre_smpl, pre_state,
                pre_cams_iter, pre_verts_iter, fut_cams_iter, fut_verts_iter, uv_feat_pre, pre_warp_feat_iter
                ):
        bs, n_frames_load, _, height, width = BP_frame_step.shape
        xfs_ref = self.x_ref(P_ref)
        x_ref = F.adaptive_avg_pool2d(xfs_ref[0], (1, 1)).view(bs, -1)  # torch.Size([1, 512])

        out_image_gen, P_pre_recoder = [], []
        x_pre = P_pre

        f_r2s = self.get_flow(cam_ref, vert_ref, pre_cams_iter, pre_verts_iter)
        pre_warp_iter = F.grid_sample(P_ref, f_r2s, align_corners=False)
        f_r2s = self.get_flow(cam_ref, vert_ref, fut_cams_iter, fut_verts_iter)
        post_warp_iter = F.grid_sample(P_ref, f_r2s, align_corners=False)

        feat_warp = []
        for i in range(n_frames_load):
            cam_step = cam_frame_step[:, i, ...]
            vert_step = vert_frame_step[:, i, ...]
            f_r2s = self.get_flow(cam_ref, vert_ref, cam_step, vert_step)
            feat_warp.append(self.warp_encoder(F.grid_sample(P_ref, f_r2s, align_corners=False)))

        if not pre_warp_feat_iter:
            pre_warp_feat_iter = self.warp_encoder(pre_warp_iter)
        for_feat_warp64, for_feat_warp128 = [], []
        feat_warp64_state = pre_warp_feat_iter[1]
        feat_warp128_state = pre_warp_feat_iter[2]
        for i in range(n_frames_load):
            temp64 = self.for_rcnn64(torch.cat((feat_warp[i][1], feat_warp64_state), dim=1))
            feat_warp64_state = temp64
            temp128 = self.for_rcnn128(torch.cat((feat_warp[i][2], feat_warp128_state), dim=1))
            feat_warp128_state = temp128
            for_feat_warp64.append(feat_warp64_state)
            for_feat_warp128.append(feat_warp128_state)
        nxt_warp_feat_iter = [None, feat_warp64_state, feat_warp128_state]

        post_warp_feat_iter = self.warp_encoder(post_warp_iter)
        post_feat_warp64, post_feat_warp128 = [], []
        post_warp64_state = post_warp_feat_iter[1]
        post_warp128_state = post_warp_feat_iter[2]
        for i in range(n_frames_load-1, -1, -1):
            temp64 = self.back_rcnn64(torch.cat((feat_warp[i][1], post_warp64_state), dim=1))
            post_warp64_state = temp64
            temp128 = self.back_rcnn128(torch.cat((feat_warp[i][2], post_warp128_state), dim=1))
            post_warp128_state = temp128
            post_feat_warp64.append(post_warp64_state)
            post_feat_warp128.append(post_warp128_state)

        for i in range(n_frames_load):
            x_pre = P_ref if x_pre is None else x_pre
            uv_feas = self.x_encoder(torch.cat((BP_frame_step[:, i], x_pre), dim=1))[0] # [torch.Size([1, 512, 16, 16]), torch.Size([1, 256, 32, 32]), torch.Size([1, 128, 64, 64]), torch.Size([1, 64, 128, 128])]

            if uv_feat_pre is None:
                uv_feat_pre = uv_feas

            x_curr = self.x_rcnn1(torch.cat((uv_feat_pre, uv_feas), dim=1))
            x_curr = self.x_rcnn2(x_curr)
            uv_feat_pre = x_curr

            x_curr = self.conv1up(x_curr, x_ref)  # 32->64
            x_forward = self.forconv1(for_feat_warp64[i], x_ref, torch.cat((x_curr, for_feat_warp64[i]), dim=1))
            x_backward = self.backconv1(post_feat_warp64[n_frames_load-1-i], x_ref, torch.cat((x_curr, post_feat_warp64[n_frames_load-1-i]), dim=1))
            x_curr = self.directconv1(x_curr, x_ref)
            x_curr = x_curr + x_forward + x_backward

            x_curr = self.conv2up(x_curr, x_ref)  # 64->128
            x_forward = self.forconv2(for_feat_warp128[i], x_ref, torch.cat((x_curr, for_feat_warp128[i]), dim=1))
            x_backward = self.backconv2(post_feat_warp128[n_frames_load-1-i], x_ref, torch.cat((x_curr, post_feat_warp128[n_frames_load-1-i]), dim=1))
            x_curr = self.directconv2(x_curr, x_ref)
            x_curr = x_curr + x_forward + x_backward

            x_curr = self.conv3up(x_curr, x_ref)  # 128->256
            x_curr = self.directconv3(x_curr, x_ref)

            x_curr = self.decoder256(x_curr, x_ref)
            image_gen = self.out(x_curr)
            x_pre = image_gen
            out_image_gen.append(image_gen)
            P_pre_recoder.append(x_pre)
        smoothed_smpl = smpl_step
        cams_smooth = cam_frame_step
        verts_smooth = vert_frame_step
        kpts_result = BP_frame_step
        kpts_smooth = BP_frame_step
        pre_state = []

        return out_image_gen, smoothed_smpl, pre_state, P_pre_recoder, cams_smooth, verts_smooth, \
               kpts_result, kpts_smooth, uv_feat_pre, nxt_warp_feat_iter

    def bilinearUP(self, small, big):
        _, _, h, w = big.shape
        return F.interpolate(small, (h, w), mode='bilinear', align_corners=False)

    def get_forward_backward_flows(self, n_frames_load, cam_frame_step, vert_frame_step, cam_ref, vert_ref):
        forward_flows, backward_flows = [], []
        for i in range(n_frames_load):
            cam_step = cam_frame_step[:, i, ...]
            vert_step = vert_frame_step[:, i, ...]

            if i == 0:
                campre = cam_ref
                vertpre = vert_ref
            else:
                campre = cam_frame_step[:, i - 1, ...]
                vertpre = vert_frame_step[:, i - 1, ...]
            forward_flows.append(self.get_flow(campre, vertpre, cam_step, vert_step))

            if i == n_frames_load - 1:
                campost = cam_ref
                vertpost = vert_ref
            else:
                campost = cam_frame_step[:, i + 1, ...]
                vertpost = vert_frame_step[:, i + 1, ...]
            backward_flows.append(self.get_flow(campost, vertpost, cam_step, vert_step))

    def get_flow(self, cam_from, vert_from, cam_to, vert_to, get_con=False):
        f2verts, fim, wim = self.renderer.render_fim_wim(cam_from, vert_from)
        f2verts = f2verts[:, :, :, 0:2]

        _, step_fim, step_wim = self.renderer.render_fim_wim(cam_to, vert_to)
        step_cond, _ = self.renderer.encode_fim(cam_to, vert_to, fim=step_fim, transpose=True)
        T = self.renderer.cal_bc_transform(f2verts, step_fim, step_wim)
        if get_con:
            return T, step_cond
        return T

    def get_mask(self, cam, vert):
        _, step_fim, step_wim = self.renderer.render_fim_wim(cam, vert)
        step_cond, _ = self.renderer.encode_fim(cam, vert, fim=step_fim, transpose=True)
        mask = morph(step_cond[:, -1:, :, :], ks=3, mode='erode')
        return mask

    def set_renderer(self, renderer=None):
        if renderer is not None:
            self.renderer = renderer

    def set_smpl_model(self, method=None):
        self.get_verts = method

    def obtain_map(self, pose_joints, H, W, sigma=6):
        device = pose_joints.get_device()
        pose_joints = (pose_joints + 1) / 2.0 * H
        N, num_kpts, _ = pose_joints.shape
        smoothed_kpts = torch.zeros([N, num_kpts, H, W], dtype=torch.float32).to(device)
        for n in range(N):
            for i in range(num_kpts):
                y = pose_joints[n, i, 0]
                x = pose_joints[n, i, 1]
                if x == 0 or y == 0:
                    continue
                xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H))
                xx = xx.to(device)
                yy = yy.to(device)
                smoothed_kpts[n, i, ...] = torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
        # smoothed_kpts = smoothed_kpts.to(device)
        return smoothed_kpts

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PoseSourceNet(BaseNetwork):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(PoseSourceNet, self).__init__()
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)


    def forward(self, source):
        feature_list=[]
        out = self.block0(source)
        feature_list.append(out)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list

