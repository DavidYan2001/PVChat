# lit_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from pathlib import Path
import shutil
import json
from utils import build_input_ids  # 导入辅助函数


class PersonalizedVideoLightningModule(pl.LightningModule):
    def __init__(self, config_path, sks_name, prefix_tokens, learning_rates, model_config):
        """
        Args:
            config_path (str): 模型配置路径。
            sks_name (str): 特殊标记名称，如 "<Bo>"。
            prefix_tokens (list): 前缀token列表。
            learning_rates (dict): 各部分学习率。
            model_config (dict): 模型相关配置。
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            config_path,
            config=self.config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 添加特殊tokens
        self.sks_tokens = [sks_name]
        self.prefix_tokens = prefix_tokens
        self.tokenizer.add_tokens(self.sks_tokens + self.prefix_tokens)
        self.model.lm.resize_token_embeddings(len(self.tokenizer))

        # 调整lm_head
        if hasattr(self.model.lm, 'lm_head'):
            old_lm_head = self.model.lm.lm_head
            self.model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(self.tokenizer),
                bias=False
            ).to(self.model.lm.device)
            self.model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data

        # 保存原始embedding
        self.orig_embeds = self.model.get_input_embeddings().weight.data.clone()

    def forward(self, input_ids, attention_mask, video, labels=None, video_idx=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video=video,
            labels=labels,
            video_idx=video_idx
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            video=batch['video'],
            labels=batch['labels'],
            video_idx=batch['video_idx']
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # 保持非个性化token的embedding不变
        with torch.no_grad():
            special_token_ids = self.tokenizer.convert_tokens_to_ids(
                self.sks_tokens + self.prefix_tokens
            )
            current_embeds = self.model.get_input_embeddings().weight.data

            keep_indices = torch.ones(current_embeds.size(0), dtype=torch.bool, device=current_embeds.device)
            keep_indices[special_token_ids] = False
            current_embeds[keep_indices] = self.orig_embeds[keep_indices]

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            video=batch['video'],
            labels=batch['labels'],
            video_idx=batch['video_idx']
        )
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # 测试步骤，根据你的需求自定义
        outputs = self.model.generate_caption(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            video=batch['video'],
            video_idx=batch['video_idx'],
            num_beams=5,
            max_new_tokens=50,
            top_k=50,
            top_p=0.95
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        question = batch['question'][0]
        self.log('test_generated_text', generated_text)
        return {'question': question, 'generated_answer': generated_text}

    def configure_optimizers(self):
        trainable_params = [
            {'params': self.model.personal_query_tokens, 'lr': self.hparams.learning_rates['personal_token_lr']},
            {'params': self.model.get_input_embeddings().weight[
                self.tokenizer.convert_tokens_to_ids(self.sks_tokens + self.prefix_tokens)],
             'lr': self.hparams.learning_rates['token_lr']},
            {'params': self.model.qformer.parameters(), 'lr': self.hparams.learning_rates['qformer_lr']},
        ]

        if self.hparams.model_config['llm']['use_lora']:
            lora_params = [p for n, p in self.model.named_parameters() if 'lora' in n.lower()]
            trainable_params.append({'params': lora_params, 'lr': self.hparams.learning_rates['lora_lr']})

        optimizer = torch.optim.AdamW(trainable_params)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # 保存额外的信息
        checkpoint['sks_tokens'] = self.sks_tokens
        checkpoint['prefix_tokens'] = self.prefix_tokens
        checkpoint['learning_rates'] = self.hparams.learning_rates
        checkpoint['model_config'] = self.hparams.model_config

    def on_load_checkpoint(self, checkpoint):
        # 你可以在这里加载额外的信息
        self.sks_tokens = checkpoint.get('sks_tokens', self.sks_tokens)
        self.prefix_tokens = checkpoint.get('prefix_tokens', self.prefix_tokens)
        self.hparams.learning_rates = checkpoint.get('learning_rates', self.hparams.learning_rates)
        self.hparams.model_config = checkpoint.get('model_config', self.hparams.model_config)
