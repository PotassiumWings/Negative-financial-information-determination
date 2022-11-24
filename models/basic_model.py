import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from configs.arguments import TrainingArguments


def max_pooling_with_mask(content: torch.LongTensor, mask: torch.LongTensor):
    # return: (b, hidden_size)
    mask = (1 - mask) * 1e4
    mask = mask.unsqueeze(-1).expand_as(content)
    result = content - mask
    return torch.max(result, axis=1)[0]


class BasicModel(nn.Module):
    def __init__(self, config: TrainingArguments):
        super(BasicModel, self).__init__()
        self.config = config

        self.out_dim = 1
        if config.loss == "CrossEntropyLoss":
            self.out_dim = 2

        self.bert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.fc = nn.Linear(config.hidden_size, self.out_dim)

    def forward(self, x1, x2):
        # 3, b
        context_text, mask_text = x1[0], x1[2]
        hidden_text = self.bert(context_text, attention_mask=mask_text)[0]  # b, max_len, hidden_size
        # print(hidden.shape)  # 8, 512, 768
        max_hs_text = self.dropout(max_pooling_with_mask(hidden_text, mask_text))

        # squeeze only if loss is BCE/BCEWithLogits
        out = self.fc(max_hs_text).squeeze(1)
        if self.config.loss == "BCELoss":
            out = F.sigmoid(out)
        return out
