# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
import os
import torch
import random
import numpy as np
from transformers import RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset

class Dataloader():
    def __init__(self, args, label_class):
        self.batch_size = args.batch_size
        self.args = args
        self.id2label = {id: label for id, label in enumerate(label_class)}
        self.label2id = {label: id for id, label in enumerate(label_class)}
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_class)


    def load_data(self, filename):
        with open(filename) as f:
            f =f.read().split('\n')

        roberta_input = []
        y_type = []
        labels = []

        for line in f:
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            roberta_input.append(self.tokenizer.encode_plus(parts[0], parts[1], max_length=self.args.max_lenth, return_tensors='pt'))
            y_type.append(0 if parts[1].split(' ')[-1] == 'Disease' else 1)
            labels.append(self.label2id[parts[2]])
        return [roberta_input, labels, y_type]

    def data_iterator(self, data, device, BATCH_NUM, shuffle=False):
        order = list(range(len(data[0])))
        if shuffle:
            random.shuffle(order)

        for i in range(BATCH_NUM):
            if i * self.batch_size < len(data[0]) < (i+1) * self.batch_size:
                batch_input_ids = [data[0][idx]['input_ids'][0] for idx in order[i * self.batch_size:]]
                batch_attention_mask = [data[0][idx]['attention_mask'][0] for idx in order[i * self.batch_size:]]
                batch_labels = [data[1][idx] for idx in order[i*self.batch_size:]]
                try:
                    y_type = [data[2][idx] for idx in order[i * self.batch_size:]]
                    batch_token_type_ids = [data[0][idx]['token_type_ids'][0] for idx in order[i * self.batch_size:]]
                except KeyError:
                    y_type = None
                    batch_token_type_ids = None
            else:
                batch_input_ids = [data[0][idx]['input_ids'][0] for idx in order[i * self.batch_size:(i+1) * self.batch_size]]
                batch_attention_mask = [data[0][idx]['attention_mask'][0] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_labels = [data[1][idx] for idx in order[i * self.batch_size:(i+1) * self.batch_size]]
                try:
                    y_type = [data[2][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                    batch_token_type_ids = [data[0][idx]['token_type_ids'][0] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                except KeyError:
                    y_type = None
                    batch_token_type_ids = None

            # 对齐
            batch_input_ids = pad_sequence(batch_input_ids, batch_first=True)
            batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True)
            batch_labels = torch.tensor(batch_labels)

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)


            if y_type:
                y_type = np.array(y_type)
                batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True)
                batch_token_type_ids = batch_token_type_ids.to(device)

            yield batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels, y_type


class KETEDataLoader():
    def __init__(self, args, label_class):
        self.batch_size = 21
        self.args = args
        self.id2label = {id: label for id, label in enumerate(label_class)}
        self.label2id = {label: id for id, label in enumerate(label_class)}
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_class)


    def load_data(self, filename):
        train_data = pd.read_csv(filename)

        roberta_input = []
        y_type = []
        labels = []

        for line in train_data.values:
            roberta_input.append(self.tokenizer.encode_plus(line[0], line[1], max_length=self.args.max_lenth, return_tensors='pt'))
            y_type.append(0 if line[1].split(' ')[-1] == 'Disease' else 1)
            labels.append(1)
            for i in range(2, 22):
                roberta_input.append(self.tokenizer.encode_plus(line[0], line[i], max_length=self.args.max_lenth, return_tensors='pt'))
                y_type.append(0 if line[i].split(' ')[-1] == 'Disease' else 1)
                labels.append(0)
        return [roberta_input, labels, y_type]

    def data_iterator(self, data, device, BATCH_NUM, shuffle=False):
        order = list(range(len(data[0])))
        shuffle = False
        if shuffle:
            random.shuffle(order)

        for i in range(BATCH_NUM):
            if i * self.batch_size < len(data[0]) < (i+1) * self.batch_size:
                batch_input_ids = [data[0][idx]['input_ids'][0] for idx in order[i * self.batch_size:]]
                batch_attention_mask = [data[0][idx]['attention_mask'][0] for idx in order[i * self.batch_size:]]
                batch_labels = [data[1][idx] for idx in order[i*self.batch_size:]]
                try:
                    y_type = [data[2][idx] for idx in order[i * self.batch_size:]]
                    batch_token_type_ids = [data[0][idx]['token_type_ids'][0] for idx in order[i * self.batch_size:]]
                except KeyError:
                    y_type = None
                    batch_token_type_ids = None
            else:
                batch_input_ids = [data[0][idx]['input_ids'][0] for idx in order[i * self.batch_size:(i+1) * self.batch_size]]
                batch_attention_mask = [data[0][idx]['attention_mask'][0] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_labels = [data[1][idx] for idx in order[i * self.batch_size:(i+1) * self.batch_size]]
                try:
                    y_type = [data[2][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                    batch_token_type_ids = [data[0][idx]['token_type_ids'][0] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                except KeyError:
                    y_type = None
                    batch_token_type_ids = None

            # 对齐
            batch_input_ids = pad_sequence(batch_input_ids, batch_first=True)
            batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True)
            batch_labels = torch.tensor(batch_labels)

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)


            if y_type:
                y_type = np.array(y_type)
                batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True)
                batch_token_type_ids = batch_token_type_ids.to(device)

            yield batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels, y_type


class MyDataSet(Dataset):
    def __init__(self, filename, args):
        raw_data = pd.read_csv(filename)
        self.data = []
        for line in raw_data.values:
            self.data.append([line[0], line[1]])
            for i in range(2, 22):
                self.data.append([line[0], line[i]])

        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_class)
        self.args = args

    def __getitem__(self, index):
        roberta_input = self.tokenizer.encode_plus(self.data[index][0],
                                                   self.data[index][1],
                                                   max_length=self.args.max_lenth,
                                                   return_tensors='pt',
                                                   padding='max_length'
                                                   )
        return roberta_input

    def __len__(self):
        return len(self.data)
