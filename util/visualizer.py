import numpy as np
import os
import ntpath
import time
from . import util
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        tb_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tb_logs')
        util.clear_dir(tb_dir)
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.loss_img_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss.svg')
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, f'web_{opt.name}')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, f'loss_log_{opt.name}.txt')
        self.eval_log_name = os.path.join(opt.checkpoints_dir, opt.name, f'eval_log_{opt.name}.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, nrow=8, size=256):
        if self.use_html: # save images to a html file
            rows = math.ceil(len(visuals) / nrow)
            if len(visuals) % nrow == 0:
                rows += 1
            cols = nrow
            cmb = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)

            row, col = 0, 0
            for label, image_numpy in visuals.items():
                if row > 0 and col == 0:        # pad for new row
                    col = 2
                cmb[row * size:row * size + size, col * size:col * size + size, :] = image_numpy
                col += 1
                if col == nrow:
                    col = 0
                    row += 1

            img_path = os.path.join(self.img_dir, 'epoch%.3d.png' % (epoch))
            util.save_image(cmb, img_path)
            # self.writer.add_image('result', cmb, global_step=epoch, dataformats='HWC')

    # errors: dictionary of error labels and values
    def plot_current_errors(self, iters, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': {}, 'legend': list(errors.keys())}
            for key in errors.keys():
                self.plot_data['Y'][key] = []
        fig, axs = plt.subplots(nrows=3, figsize=(8, 7), constrained_layout=True)
        titles = ['Generator', 'Gan', 'GanVideo']
        self.plot_data['X'].append(iters)
        for legend in self.plot_data['legend']:
            prefix = 0  #
            if 'video' in legend:
                prefix = 2 # 'Video'
            elif 'gan' in legend:
                prefix = 1  # 'GAN'

            self.writer.add_scalar(f'{prefix}/{legend}', errors[legend], iters)

            self.plot_data['Y'][legend].append(errors[legend])
            axs[prefix].plot(self.plot_data['X'], self.plot_data['Y'][legend],
                             label="{}-{:.3f}".format(legend, errors[legend]))
            axs[prefix].set_title(titles[prefix])
            axs[prefix].grid(True)
            if prefix == 0:
                axs[prefix].set_ylim([0, 5])
                axs[prefix].legend(loc='lower left')
            else:
                axs[prefix].legend(loc='upper right')
        fig.suptitle(self.name)
        fig.supxlabel('Iterations')
        fig.supylabel('Loss')
        plt.savefig(self.loss_img_path)
        plt.cla()
        plt.close(fig)

    def plot_current_score(self, iters, scores):
        if not hasattr(self, 'plot_score'):
            self.plot_score = {'X':[],'Y':[], 'legend':list(scores.keys())}
        self.plot_score['X'].append(iters)
        self.plot_score['Y'].append([scores[k] for k in self.plot_score['legend']])
        # self.vis.line(
        #     X=np.stack([np.array(self.plot_score['X'])] * len(self.plot_score['legend']), 1),
        #     Y=np.array(self.plot_score['Y']),
        #     opts={
        #         'title': self.name + ' Evaluation Score over time',
        #         'legend': self.plot_score['legend'],
        #         'xlabel': 'iters',
        #         'ylabel': 'score'},
        #     win=self.display_id + 29
        # )

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_left, max_iteration):
        mm, ss = divmod(t_left, 60)
        hh, mm = divmod(mm, 60)
        s = "%d:%02d:%02d" % (hh, mm, ss)
        message = '(epoch: %d, iters: %d/%d, time: %.1fs|%s) ' % (epoch, i, max_iteration, t, s)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_eval(self, epoch, i, score):
        message = '(epoch: %d, iters: %d)' % (epoch, i)
        for k, v in score.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.eval_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
