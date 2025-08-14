# datamodule.py

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import PersonalizedVideoDataset  # 确保导入正确
from dataset import collate_fn  # 如果 collate_fn 定义在 dataset.py 中
# 如果 collate_fn 定义在 utils.py 中，改为:
# from utils import collate_fn

class PersonalizedVideoDataModule(pl.LightningDataModule):
    def __init__(self, data_root, sks_name, tokenizer, batch_size=2, num_workers=4):
        super().__init__()
        self.data_root = Path(data_root)
        self.sks_name = sks_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_json_path = self.data_root / f"{self.sks_name}.json"
        test_json_path = self.data_root / f"{self.sks_name}test.json"

        self.train_dataset = PersonalizedVideoDataset(
            json_path=str(train_json_path),
            tokenizer=self.tokenizer,
            split='train'
        )

        self.test_dataset = PersonalizedVideoDataset(
            json_path=str(test_json_path),
            tokenizer=self.tokenizer,
            split='test'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        # 如果有验证集，可以在这里定义
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # 单个问题逐个测试
            shuffle=False,
            collate_fn=collate_fn
        )
