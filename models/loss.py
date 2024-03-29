import torch
import torch.nn as nn

from configs.arguments import TrainingArguments


class Loss:
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.label_smoothing = config.label_smoothing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gap = 0
        self.prompt_loss_part = self.config.prompt_loss
        if self.config.prompt:
            self.loss = self.prompt_loss
        elif self.config.loss == "BCELoss":
            self.gap = 0.5
            self.loss = self.bce_loss
        elif self.config.loss == "BCEWithLogitsLoss":
            self.loss = self.bce_loss_logits
        elif self.config.loss == "CrossEntropyLoss":
            self.loss = self.ce_loss
        else:
            raise NotImplementedError(f"This loss {self.config.loss} is not implemented.")

        if self.config.judge_gap != 1234:
            self.gap = self.config.judge_gap

    def get_loss(self):
        return self.loss

    def bce_loss(self, output, labels):
        return nn.BCELoss()(output, labels * (1 - 2 * self.label_smoothing) + self.label_smoothing)

    def bce_loss_logits(self, output, labels):
        return nn.BCEWithLogitsLoss()(output, labels * (1 - 2 * self.label_smoothing) + self.label_smoothing)

    def ce_loss(self, output, labels):
        return nn.CrossEntropyLoss()(output, labels)

    def prompt_loss(self, output, labels):
        # output: positive b * 1, negative b * 1
        negative, positive = output
        if self.prompt_loss_part == "max":
            negative = torch.max(negative, dim=1)[0]
            positive = torch.max(positive, dim=1)[0]
        else:
            assert self.prompt_loss_part == "sum", self.prompt_loss_part
            negative = torch.sum(negative, dim=1)
            positive = torch.sum(positive, dim=1)

        all_output = torch.cat([negative.unsqueeze(-1), positive.unsqueeze(-1)], 1)
        return nn.CrossEntropyLoss()(all_output, labels).to(self.device)
