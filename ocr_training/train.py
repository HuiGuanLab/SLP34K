#!/usr/bin/env python3
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
import os
from pathlib import Path

from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

import math
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize
import pytorch_lightning as pl
from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


import torch
import numpy as np
import random

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def readfile(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read().splitlines()
    return ''.join(content)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):

    seed = 42
    pl.seed_everything(seed, workers=True)

    if os.path.isfile(config.model.charset_train):
        config.model.charset_train = readfile(config.model.charset_train)
        config.model.charset_test = config.model.charset_train
    
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpus = config.trainer.get('gpus', 0)
        if gpus:
            # Use mixed-precision training
            config.trainer.precision = 16
        if gpus > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True) # Check whether each parameter participates in loss calculation, so there is extra overhead
            # trainer_strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True) 
            # Scale steps-based config
            config.trainer.val_check_interval //= gpus  
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= gpus

    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
   
    print(summarize(model, 2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')

    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint]
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               accumulate_grad_batches=config.trainer.accumulate_grad_batches,
                                               callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    
if __name__ == '__main__':
    main()
