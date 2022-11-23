from configs.arguments import TrainingArguments
import torch.nn as nn


class Loss:
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.label_smoothing = config.label_smoothing
        self.gap = 0
        if self.config.loss == "BCELoss":
            self.gap = 0.5
            self.loss = self.bce_loss
        elif self.config.loss == "BCEWithLogitsLoss":
            self.loss = self.bce_loss_logits
        elif self.config.loss == "CrossEntropyLoss":
            self.loss = self.ce_loss
        raise NotImplementedError(f"This loss {self.config.loss} is not implemented.")

    def get_loss(self):
        return self.loss

    def bce_loss(self, output, labels):
        return nn.BCELoss()(output, labels * (1 - 2 * self.label_smoothing) + self.label_smoothing)

    def bce_loss_logits(self, output, labels):
        return nn.BCEWithLogitsLoss()(output, labels * (1 - 2 * self.label_smoothing) + self.label_smoothing)

    def ce_loss(self, output, labels):
        return nn.CrossEntropyLoss()(output, labels)
