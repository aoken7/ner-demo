'''
Copyright (c) 2021 Stockmark Inc.
Released under the MIT license
https://opensource.org/licenses/mit-license.php
'''


import itertools
import random
import json
from typing import Dict
from tqdm import tqdm
import numpy as np
import unicodedata
import pandas as pd

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import pytorch_lightning as pl

import ner_tokenizer_BIO

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

type_id_dict = {
    "人名": 1,
    "法人名": 2,
    "政治的組織名": 3,
    "その他の組織名": 4,
    "地名": 5,
    "施設名": 6,
    "製品名": 7,
    "イベント名": 8
}


class BertForTokenClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class Inference():
    def __init__(self) -> None:
        self.tokenizer = ner_tokenizer_BIO.NER_tokenizer_BIO.from_pretrained(
            MODEL_NAME,
            num_entity_type=len(type_id_dict)
        )
        self.model = BertForTokenClassification_pl.load_from_checkpoint(
            '/working_dir/backend/models/epoch=8-step=909-v2.ckpt'
        )
        self.bert_tc = self.model.bert_tc.cuda()

    def inference(self, text: str) -> Dict:

        encoding, spans = self.tokenizer.encode_plus_untagged(
            text, return_tensors='pt'
        )
        encoding = {k: v.cuda() for k, v in encoding.items()}

        with torch.no_grad():
            output = self.bert_tc(**encoding)
            scores = output.logits
            scores = scores[0].cpu().numpy().tolist()

        # 分類スコアを固有表現に変換する
        entities_predicted = self.tokenizer.convert_bert_output_to_entities(
            text, scores, spans
        )

        return entities_predicted


if __name__ == '__main__':
    text = '日本の首都は東京都で、総理大臣は岸田総理です。'
    inference = Inference()
    print(inference.inference(text))
