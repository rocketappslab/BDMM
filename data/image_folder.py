###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

POSE_EXTENSIONS = [
    '.txt', '.json','.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_pose_file(filename):
    return any(filename.endswith(extension) for extension in POSE_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images




def default_loader(path):
    return Image.open(path).convert('RGB')



class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

# add for face dataset

def make_grouped_dataset(dir, ext, subdir):
    file_ext = is_image_file if ext == 'img' else is_pose_file
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            file = os.path.join(root, f)
            if file_ext(f) and f'/{subdir}/' in file:
                paths.append(file)
        if len(paths) > 0:
            images.append(paths)
    return images

def check_path_valid(A_paths, B_paths):
    assert(len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert(len(a) == len(b))