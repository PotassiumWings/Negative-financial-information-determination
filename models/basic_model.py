import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from configs.arguments import TrainingArguments


def max_pooling_with_mask(content: torch.LongTensor, mask: torch.LongTensor):
    # return: (b, hidden_size)
    mask = (1 - mask) * 1e6  # TODO
    mask = mask.unsqueeze(-1).expand_as(content)
    result = content - mask
    return torch.max(result, axis=1)[0]


class BasicModel(nn.Module):
    def __init__(self, config: TrainingArguments):
        super(BasicModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.fc = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        # 3, b
        context = x[0]  # b
        mask = x[2]  # b
        hidden = self.bert(context, attention_mask=mask)[0]  # b, max_len, hidden_size  TODO
        # print(hidden.shape)  # 8, 512, 768
        max_hs = max_pooling_with_mask(hidden, mask)
        hs = self.dropout(max_hs)
        out = self.fc(hs)
        # out = F.softmax(out, dim=1)
        return out
