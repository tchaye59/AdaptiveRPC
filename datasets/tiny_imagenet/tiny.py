# Adapted from https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54

import imageio
import numpy as np
import os

from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.auto import tqdm

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

import requests, zipfile
from PIL import Image, ImageOps


def download_and_unzip(URL, root_dir):
    if os.path.exists(os.path.join(root_dir, "tiny-imagenet-200")):
        return
    os.makedirs(root_dir, exist_ok=True)
    path = os.path.join(root_dir, "tiny-imagenet-200.zip")
    # Download
    r = requests.get(URL, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        n_chunks = (total_length / 1024) + 1
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=n_chunks):
            if chunk:
                f.write(chunk)
                f.flush()
    # Unzip
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    # Delete the zip file
    os.remove(path)


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, "tiny-imagenet-200", 'train')
        val_path = os.path.join(root_dir, "tiny-imagenet-200", 'val')
        test_path = os.path.join(root_dir, "tiny-imagenet-200", 'test')

        wnids_path = os.path.join(root_dir, "tiny-imagenet-200", 'wnids.txt')
        words_path = os.path.join(root_dir, "tiny-imagenet-200", 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None, sub_targets=-1):
        tinp = TinyImageNetPaths(root_dir, download)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        if sub_targets > 1:
            self.samples = [s for s in self.samples if s[self.label_idx] < sub_targets]
        self.samples_num = len(self.samples)
        self.targets = [s[self.label_idx] for s in self.samples]

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                     dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]

                # img = Image.open(s[0])
                # if img.layers != 1:
                #    img = ImageOps.grayscale(img)
                # img = np.array(img)[:, :, None]

                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx].astype(np.uint8)
            # img = np.squeeze(img, axis=-1)
            img = Image.fromarray(img)
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = Image.open(s[0])
            # if img.layers != 1:
            #    img = ImageOps.grayscale(img)
            # img = imageio.imread(s[0])
            lbl = None if self.mode == 'test' else s[self.label_idx]
        # sample = {'image': img, 'label': lbl}

        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)
        return img, torch.tensor(lbl, dtype=torch.long)


if __name__ == "__main__":
    ds = TinyImageNetDataset('tiny_ds', mode='train', download=True, sub_targets=10)
    img_size = ds[0][0].shape
    print(img_size)
    train_loader = DataLoader(ds, batch_size=32, shuffle=False)
    i = 100
    for x in train_loader:
        print(x[0].shape)
        i -= 1
        if i <= 0:
            break
    pass
