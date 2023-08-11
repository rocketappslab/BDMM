# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import pickle
import toml
import torch
import numpy as np
import codecs, json

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

    return paths


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def clear_dir(path):
    import shutil
    if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
        shutil.rmtree(path)
    path = mkdir(path)

    return path


def remove_dir(path):
    import shutil
    if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
        shutil.rmtree(path)


def load_txt_file(txt_path):
    paths = []
    with open(txt_path) as f:
        for line in f:
            paths.append(line.rstrip('\n'))
    return paths


def load_pickle_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, "wb") as fp:
        pickle.dump(data_dict, fp, protocol=2)


def load_json_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json_file(json_file, data_dict):
    with open(json_file, "w") as f:
        json_str = json.dumps(data_dict)

        f.writelines(json_str)

def jsonify(json_file, np_dict):
    new_dict = {}
    for k, v in np_dict.items():
        if type(v) is not list:
            new_dict[k] = v.tolist()
        else:
            new_dict[k] = v

    json.dump(new_dict, codecs.open(json_file, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)


def unjsonify(json_file):
    obj_text = codecs.open(json_file, 'r', encoding='utf-8').read()
    data = json.loads(obj_text)
    return data


def load_toml_file(toml_file):

    with open(toml_file, "r", encoding="utf-8") as f:
        data = toml.load(f)

    return data


def write_toml_file(toml_file, data_dict):
    with open(toml_file, "w", encoding="utf-8") as fp:
        toml.dump(data_dict, fp)

def image_to_tensor(input, device):
    image = input.copy()
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float) / 255
    image = torch.tensor(image).float()[None].to(device)
    return image

def auto_unzip_fun(x, f):
    return f(*x)
