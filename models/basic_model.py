import torch.nn as nn
import torch
import torch.nn.functional as F
from configs.basic_config_bert import BasicConfig


class BasicModel(nn.Module):
    def __init__(self, config: BasicConfig):
        super(BasicModel, self).__init__()
        self.bert = config.model
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        # 3, b
        context = x[0]  # b
        mask = x[2]  # b
        hidden = self.bert(context, attention_mask=mask)[0]  # b, max_len, hidden_size  TODO
        print(hidden.shape)
        max_hs = self.maxpooling(hidden, mask)
        hs = self.dropout(max_hs)
        out = F.softmax(self.fc(hs), dim=1)
        return out

