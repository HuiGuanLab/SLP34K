# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import PurePath
from typing import Optional, Callable, Sequence, Tuple
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .dataset import build_tree_dataset, LmdbDataset
import natsort,glob
class SceneTextDataModule(pl.LightningDataModule):
    TEST_Six_BENCHMARK = ['IIIT5k', 'SVT','IC13_1015','IC15_2077' ,'SVTP', 'CUTE80']
    TEST_UNION14M_BENCHMARK = ['curve','multi_oriented','artistic','contextless','salient','multi_words','general']
    TEST_ALL_COMMON = list(TEST_Six_BENCHMARK+TEST_UNION14M_BENCHMARK)
    TEST_SLP34K = natsort.natsorted(glob.glob('./data/test/SLP34K_lmdb_benchmark/*_lmdb'))
    TEST_SLP34K = [os.path.basename(x) for x in TEST_SLP34K]

    def __init__(self, root_dir: str, train_dir: str, val_dir: str,test_dir: str,img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, augment: bool,
                 remove_whitespace: bool = False, normalize_unicode: bool = False,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None
        self.mean = (0.48145466, 0.4578275, 0.40821073) 
        self.std = (0.26862954, 0.26130258, 0.27577711) 
    
    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0, mean=0.5, std=0.5):
        transforms = []
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return T.Compose(transforms)
    
    
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            root = PurePath(self.root_dir, 'train', self.train_dir)
            self._train_dataset = build_tree_dataset(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            root = PurePath(self.root_dir, 'val',self.val_dir)
            self._val_dataset = build_tree_dataset(root, self.charset_test, self.max_label_length,
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   transform=transform)
        return self._val_dataset
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)
    

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir,'test',self.test_dir)
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
