from options.test_options import TestOptions
import data as Dataset
from model import create_model
import torch
from tqdm import tqdm
if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset), total=dataset_size):
            model.set_input(data)
            model.test(i, opt.print_frq)


