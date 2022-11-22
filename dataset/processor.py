import csv
import random

import torch
import tqdm
from numpy import ceil
from transformers import BertTokenizer

from configs.arguments import TrainingArguments


class Dataset:
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)

        self.f_max_seq_len = self.config.f_max_seq_len
        self.t_max_seq_len = self.config.max_seq_len - self.f_max_seq_len

        train_file_path = config.train_file
        test_file_path = config.test_file
        self.max_seq_len = config.max_seq_len
        self.train_data = self._load_data(train_file_path, is_test=False)
        self.test_data = self._load_data(test_file_path, is_test=True)
        self.train_iter, self.val_iter = self._get_train_val_data_iter()
        self.test_iter = self._get_test_data_iter()

    def _load_data(self, filepath, is_test=False):
        contents = []
        with open(filepath, 'r', encoding="GB18030") as f:
            sr = csv.reader(f)
            for row in tqdm.tqdm(sr):
                # first row
                if row[0] == 'id':
                    if is_test:
                        # assert row[1] == 'text' and row[2] == 'entity'
                        assert row[2] == 'text' and row[3] == 'entity'
                    else:
                        # assert row[1] == 'text' and row[2] == 'entity' and row[3] == 'negative'
                        assert row[2] == 'text' and row[3] == 'entity' and row[4] == 'negative'
                    continue

                if is_test:
                    # id, text, entities
                    label = row[0]
                else:
                    # id, text, entities, label, entities_real
                    label = int(row[4])

                # only text
                text = row[1]
                token_ids = self.tokenizer.encode(text, add_special_tokens=self.config.add_special_tokens)
                seq_len = len(token_ids)
                if seq_len < self.max_seq_len:
                    mask = [1] * len(token_ids) + [0] * (self.max_seq_len - seq_len)
                    token_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - seq_len)
                else:
                    mask = [1] * self.max_seq_len
                    # token_ids = token_ids[:self.max_seq_len]
                    token_ids = token_ids[:self.config.f_max_seq_len] + token_ids[seq_len - self.config.t_max_seq_len:]
                    seq_len = self.max_seq_len
                contents.append((token_ids, label, seq_len, mask))
        return contents

    def _get_test_data_iter(self):
        contents = self.test_data
        return DatasetIterator(contents, self.config.batch_size, self.device)

    def _get_train_val_data_iter(self):
        contents = self.train_data
        train_contents = contents[:int(self.config.train_percent * len(contents))]
        valid_contents = contents[int(self.config.train_percent * len(contents)):]
        return DatasetIterator(train_contents, self.config.batch_size, self.device), \
               DatasetIterator(valid_contents, self.config.batch_size, self.device)


class DatasetIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        random.shuffle(self.batches)
        self.n_batches = int(ceil(len(batches) // batch_size))
        self.index = 0
        self.device = device

    def shuffle(self):
        random.shuffle(self.batches)

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.FloatTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                      self.index * self.batch_size: min((self.index + 1) * self.batch_size, len(self.batches))]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches
