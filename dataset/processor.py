import csv
import random
from functools import reduce

import torch
import tqdm
from numpy import ceil
from transformers import AutoTokenizer

from configs.arguments import TrainingArguments


class Dataset:
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.f_max_seq_len = self.config.f_max_seq_len
        self.t_max_seq_len = self.config.max_seq_len - self.f_max_seq_len

        train_file_path = config.data_path + config.data_prefix + config.train_file
        test_file_path = config.data_path + config.data_prefix + config.test_file
        self.max_seq_len = config.max_seq_len

        mask_token = self.tokenizer.encode(self.config.mask_str, add_special_tokens=False)
        assert len(mask_token) == 1, mask_token
        self.mask_token = mask_token

        # test
        self.test_labels = []  # row[0]
        self.max_index = 0  # 2;小资钱包 <-> 5
        self.label_to_index, self.index_to_label = {}, {}

        if self.config.model_filename == "" or self.config.fine_tune:
            self.train_data = self._load_data(train_file_path, is_test=False)
            self.train_iter, self.val_iter = self._get_train_val_data_iter()
        self.test_data = self._load_data(test_file_path, is_test=True)
        self.test_iter = self._get_test_data_iter()

    def _load_data(self, filepath, is_test=False):
        contents = []
        with open(filepath, 'r', encoding="GB18030") as f:
            sr = csv.reader(f)
            for row in tqdm.tqdm(sr):
                # first row
                if row[0] == 'id':
                    if is_test:
                        assert row[2] == 'text' and row[3] == 'entity'
                    else:
                        assert row[2] == 'text' and row[3] == 'entity' \
                               and row[4] == 'negative' and row[5] == 'key_entity'
                    continue

                entities = row[3].split(";")
                if is_test:
                    self.test_labels.append(row[0])
                else:
                    key_entities = row[5].split(";")

                for i in range(len(entities)):
                    entity = entities[i]
                    text = row[2]
                    # replace entities that do not contain `entity` and not contained by `entity` to `其他实体`
                    if self.config.replace_entity:
                        text = reduce(
                            lambda x, y: x.replace(y, "其他实体"),
                            filter(lambda x: x not in entity and entity not in x, entities),
                            text
                        )

                    if self.config.prompt:
                        if self.config.prompt_pattern == 0:
                            text = text + "总之，" + entity + self.config.mask_str
                        elif self.config.prompt_pattern == 1:
                            text = entity + self.config.mask_str + "！" + text
                        else:
                            assert self.config.prompt_pattern == 2
                            text = entity + "不好" + self.config.mask_str + text
                    else:
                        text = text + "上文主要评论的是" + entity

                    text_token_ids, text_seq_len, text_mask, prompt_pos = self._get_token_ids(text, require_prompt=True)
                    entity_token_ids, entity_seq_len, entity_mask, _ = self._get_token_ids(entity)
                    if is_test:
                        # id, title, text, entity
                        label = f"{len(self.test_labels) - 1};{entity}"
                        if label not in self.label_to_index:
                            self.label_to_index[label] = self.max_index
                            self.index_to_label[self.max_index] = label
                            self.max_index += 1
                        label = self.label_to_index[label]
                    else:
                        # id, title, text, entity, negative, key_entity
                        label = reduce(lambda x, y: x or y == entity, key_entities, False)

                    # temporarily separate text and entity
                    contents.append((text_token_ids, text_seq_len, text_mask, prompt_pos,
                                     entity_token_ids, entity_seq_len, entity_mask, label))
        return contents

    def _get_token_ids(self, text, require_prompt=False):
        token_ids = self.tokenizer.encode(text, add_special_tokens=self.config.add_special_tokens)
        seq_len = len(token_ids)
        if seq_len < self.max_seq_len:
            mask = [1] * len(token_ids) + [0] * (self.max_seq_len - seq_len)
            token_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - seq_len)
        else:
            mask = [1] * self.max_seq_len
            # token_ids = token_ids[:self.max_seq_len]
            token_ids = token_ids[:self.f_max_seq_len] + token_ids[seq_len - self.t_max_seq_len:]
            seq_len = self.max_seq_len

        prompt_pos = -1
        if require_prompt:
            # mask_token appears, and only appears once
            prompt_pos = token_ids.index(self.mask_token)
            try:
                token_ids[prompt_pos + len(str(self.mask_token)):].index(self.mask_token)
                assert False
            except ValueError:
                pass

        return token_ids, seq_len, mask, prompt_pos

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
        text_x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        text_seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        text_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        prompt_pos = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        entity_x = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        entity_seq_len = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        entity_mask = torch.LongTensor([_[6] for _ in datas]).to(self.device)

        y = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        return (text_x, text_seq_len, text_mask, prompt_pos), (entity_x, entity_seq_len, entity_mask), y

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
