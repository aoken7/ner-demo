'''
Copyright (c) 2021 Stockmark Inc.
Released under the MIT license
https://opensource.org/licenses/mit-license.php
'''


import itertools
import random
import json
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

pl.seed_everything(1)

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

dataset = json.load(open('/working_dir/backend/dataset/ner.json', 'r'))


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


def normalization(dataset):
    # カテゴリーをラベルに変更、文字列の正規化する。
    for sample in dataset:
        sample['text'] = unicodedata.normalize('NFKC', sample['text'])
        for e in sample["entities"]:
            e['type_id'] = type_id_dict[e['type']]
            del e['type']
    return dataset


# データセットの分割
random.shuffle(dataset)
n = len(dataset)
n_train = int(n*0.6)
n_val = int(n*0.2)
dataset_train = dataset[:n_train]
dataset_val = dataset[n_train:n_train+n_val]
dataset_test = dataset[n_train+n_val:]
# random.shuffle(dataset_train)
# random.shuffle(dataset_val)
# random.shuffle(dataset_test)

dataset_train = normalization(dataset_train)
dataset_val = normalization(dataset_val)
dataset_test = normalization(dataset_test)


def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
    return dataset_for_loader


# PyTorch Lightningのモデル
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


def predict(text, tokenizer, bert_tc):
    """
    BERTで固有表現抽出を行うための関数。
    """
    # 符号化
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    )
    encoding = {k: v.cuda() for k, v in encoding.items()}

    # ラベルの予測値の計算
    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()

    # ラベル列を固有表現に変換
    entities = tokenizer.convert_bert_output_to_entities(
        text, labels_predicted, spans
    )

    return entities


def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0  # 固有表現(正解)の個数
    num_predictions = 0  # BERTにより予測された固有表現の個数
    num_correct = 0  # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較。
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。

    for entities, entities_predicted \
            in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [e for e in entities if e['type_id'] == type_id]
            entities_predicted = [
                e for e in entities_predicted if e['type_id'] == type_id
            ]

        def get_span_type(e): return (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set(get_span_type(e) for e in entities)
        set_entities_predicted = \
            set(get_span_type(e) for e in entities_predicted)

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len(set_entities & set_entities_predicted)

    # 指標を計算
    if num_predictions == 0:
        precision = -1
    else:
        precision = num_correct/num_predictions  # 適合率
    recall = num_correct/num_entities  # 再現率
    if precision+recall == 0:
        f_value = -1
    else:
        f_value = 2*precision*recall/(precision+recall)  # F値

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result


tokenizer = ner_tokenizer_BIO.NER_tokenizer_BIO.from_pretrained(
    MODEL_NAME,
    num_entity_type=len(type_id_dict)
)


# データセットの作成
max_length = 128
dataset_train_for_loader = create_dataset(
    tokenizer, dataset_train, max_length
)
dataset_val_for_loader = create_dataset(
    tokenizer, dataset_val, max_length
)

# データローダの作成
dataloader_train = DataLoader(
    dataset_train_for_loader, batch_size=32, shuffle=False
)
dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)


# ファインチューニング

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='models'
)


trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    deterministic=True,
    callbacks=[checkpoint]
)

# PyTorch Lightningのモデルのロード
num_entity_type = len(type_id_dict)
num_labels = 2*num_entity_type+1
model = BertForTokenClassification_pl(
    MODEL_NAME, num_labels=num_labels, lr=1e-5
)

# ファインチューニング
trainer.fit(model, dataloader_train, dataloader_val)
best_model_path = checkpoint.best_model_path

torch.save(model.state_dict(), '/working_dir/backend/models/checkpointbest.pth')

# 性能評価
model = BertForTokenClassification_pl.load_from_checkpoint(
    best_model_path
)

bert_tc = model.bert_tc.cuda()

entities_list = []  # 正解の固有表現を追加していく
entities_predicted_list = []  # 抽出された固有表現を追加していく
for sample in tqdm(dataset_test):
    text = sample['text']
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    )
    encoding = {k: v.cuda() for k, v in encoding.items()}

    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        scores = scores[0].cpu().numpy().tolist()

    # 分類スコアを固有表現に変換する
    entities_predicted = tokenizer.convert_bert_output_to_entities(
        text, scores, spans
    )

    entities_list.append(sample['entities'])
    entities_predicted_list.append(entities_predicted)


print(evaluate_model(entities_list, entities_predicted_list))
print(best_model_path)
