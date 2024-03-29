import json
import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from configs.arguments import TrainingArguments


class PromptModel(nn.Module):
    def __init__(self, config: TrainingArguments):
        super(PromptModel, self).__init__()
        self.config = config
        self.bert = AutoModelForMaskedLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.vocabs = json.load(open(os.path.join(self.config.model_name, "tokenizer.json"), "r", encoding="utf-8"))

        self.prompt_positives = config.prompt_positive
        self.prompt_negatives = config.prompt_negative
        self.prompt_positives_indexes = self._get_index(self.prompt_positives)
        self.prompt_negatives_indexes = self._get_index(self.prompt_negatives)
        assert len(self.prompt_negatives) == len(self.prompt_positives)
        logging.info(f"[MASK]: {self.tokenizer.encode('[MASK]', add_special_tokens=False)},"
                     f" <mask>: {self.tokenizer.encode('<mask>', add_special_tokens=False)}")
        logging.info(f"Prompt: + {self.prompt_positives_indexes} - {self.prompt_negatives_indexes}")

    def _get_index(self, s):
        # result = [self.vocabs["model"]["vocab"][ch] for ch in s]
        if "xlm-roberta" in self.config.model_name:
            result = []
            for ch in s:
                sub_result = self.tokenizer.encode(ch, add_special_tokens=False)
                if sub_result[0] == 6 and len(sub_result) == 2:
                    result.append(sub_result[1])
                else:
                    # ？！
                    sub_result = self.tokenizer.encode("测" + ch, add_special_tokens=False)
                    assert sub_result[0] == 6 and len(sub_result) == 3, sub_result
                    result.append(sub_result[2])
        else:
            result = [self.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in s]
        return result

    def forward(self, x1, x2):
        # b * 512, b * 512, b
        context_text, mask_text, prompt_pos = x1[0], x1[2], x1[3]
        predict = self.bert(context_text, attention_mask=mask_text)[0]  # b, max_len, vocab_size
        positive_result, negative_result = [], []
        for i in range(self.config.batch_size):
            positive_result.append(predict[i][prompt_pos[i]][self.prompt_positives_indexes])
            negative_result.append(predict[i][prompt_pos[i]][self.prompt_negatives_indexes])
        # b * len(prompt_positive)
        pos_tensor, neg_tensor = torch.stack(positive_result), torch.stack(negative_result)
        return neg_tensor, pos_tensor
