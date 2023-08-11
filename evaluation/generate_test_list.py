import pandas as pd
import ast

def read_gflacsv(file):
    file = pd.read_csv(file)
    # A_paths = file['A_paths'].map(ast.literal_eval)
    data = {
        "A_paths": [],
        "B_paths_clean": []
    }
    A_paths = file['A_paths'].map(ast.literal_eval)
    for record in A_paths:
        video, ref = record['ref'][0].split('/')[-2:]
        img_ref = ['./dataset/UBC_fashion_smpl/test/' + video + '.mp4/frames/frame_' + ref]
        img_frames = ['./dataset/UBC_fashion_smpl/test/' + video + '.mp4/frames/frame_' + f.split('/')[-1] for f in record['gen']]
        img_d = {
            'ref': img_ref,
            'gen': img_frames
        }

        kpts = ['./dataset/UBC_fashion_smpl/test/' + video + '.mp4/kptsmpls/frame_' + f.split('/')[-1][:-4] + '.json' for f in record['gen']]
        kpt_ref = ['./dataset/UBC_fashion_smpl/test/' + video + '.mp4/kptsmpls/frame_' + ref[:-4] + '.json']
        kpt_d = {
            'ref': kpt_ref,
            'gen': kpts
        }

        data['A_paths'].append(img_d)
        data['B_paths_clean'].append(kpt_d)

    df = pd.DataFrame(data)
    df.to_csv('test_list.csv', index=True)


if __name__=='__main__':
    read_gflacsv(file='../Dataset/danceFashion/test_list.csv')