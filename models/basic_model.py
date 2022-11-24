import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

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

        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.fc = nn.Linear(config.hidden_size, self.out_dim)
        # self.fc = nn.Linear(config.hidden_size * 2, self.out_dim)

    def forward(self, x1, x2):
        # 3, b
        context_text, mask_text = x1[0], x1[2]
        # context_entity, mask_entity = x2[0], x2[2]
        # use same bert  TODO
        hidden_text = self.bert(context_text, attention_mask=mask_text)[0]  # b, max_len, hidden_size
        # hidden_entity = self.bert(context_entity, attention_mask=mask_entity)[0]  # b, max_len, hidden_size
        # print(hidden.shape)  # 8, 512, 768
        # use same dropout  TODO
        max_hs_text = self.dropout(max_pooling_with_mask(hidden_text, mask_text))
        # max_hs_entity = self.dropout(max_pooling_with_mask(hidden_entity, mask_entity))

        # squeeze only if loss is BCE/BCEWithLogits
        out = self.fc(max_hs_text).squeeze()  # TODO: batch_size = 1, squeeze bad
        # out = self.fc(torch.cat([max_hs_text, max_hs_entity], dim=-1)).squeeze()
        if self.config.loss == "BCELoss":
            out = F.sigmoid(out)
        return out
