from pathlib import Path
from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
import os
import mxnet as mx
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm

def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        print(type(img))
        label = int(header.label[0])
        
        # 원본 소스: 이미지 객체로 변환후 저장 -> bgr 채널로 저장
#         img = Image.fromarray(img)  # numpy 배열을 Image객체로 바꿀 때
#         label_path = save_path/str(label)
#         if not label_path.exists():
#             label_path.mkdir()
#         img.save(label_path/'{}.jpg'.format(idx), quality=95)

        # openCV로 numpy로 저장 -> rgb 채널순으로 저장
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        cv2.imwrite(f'{label_path}/{idx}.jpg',img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():

	conf = edict()
	conf.data_path = Path('../data')

	args = edict()
	# args.rec_path = '../data/small_vgg/'
	args.rec_path = 'my_webface'
	rec_path = conf.data_path/args.rec_path

	print(rec_path, os.path.isdir(rec_path))

	load_mx_rec(rec_path)

if __name__ == '__main__':
	main()

