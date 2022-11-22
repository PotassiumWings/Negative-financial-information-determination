from torch import nn
from torch.optim import Adam
from configs.basic_config_bert import BasicConfig
import torch
import logging
import torch.nn.functional as F


class Trainer:
    def __init__(self, config: BasicConfig, model: nn.Module, dataset):
        self.model = model
        self.dataset = dataset
        self.config = config
        # loss: (b, 1) (b, 1) -> num
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def train(self):
        train_iter, val_iter = self.dataset.train_iter, self.dataset.val_iter
        # set state
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        best_val_loss = float('inf')
        last_improve = 0  # last time to update best_val_loss
        current_batch = 0

        for epoch in range(self.config.num_epoches):
            train_iter.shuffle()
            trues, predicts = [], []
            # texts: [tensor[8, 512], tensor[8, 512], tensor[8, 512]]  x, seq_len, mask
            # labels: tensor[8]
            for i, (texts, labels) in enumerate(train_iter):
                # outputs: [8, 2]
                outputs = self.model(texts)
                self.model.zero_grad()
                loss = torch.sum(self.loss(outputs, labels))
                loss.backward()
                optimizer.step()

                trues.append(labels.cpu())
                predicts.append(outputs.cpu())

                if current_batch % self.config.show_period == 0:
                    # output accuracy
                    train_acc = self.calc_train_acc(trues, predicts)
                    val_acc, val_loss = self.eval(val_iter)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), self.config.save_path)
                        last_improve = current_batch
                    logging.log(f"Ep {epoch}/{self.config.num_epoches}, iter {current_batch},"
                                f" train loss {loss.item()}, train acc {train_acc},"
                                f" val loss {val_loss}, val acc {val_acc}, last upd {last_improve}")
                    self.model.train()
                current_batch += 1
                if current_batch - last_improve > self.config.early_stop_diff:
                    logging.info("No improve for long, early stopped.")
                    return

    def eval(self, val_iter):
        self.model.eval()
        total_loss = 0
        trues, predicts = [], []
        with torch.no_grad():
            for i, (texts, labels) in enumerate(val_iter):
                outputs = self.model(texts)
                loss = self.loss(outputs, labels)
                total_loss += torch.sum(loss).item()

                trues.append(labels.cpu())
                predicts.append(outputs.cpu())

        val_acc = self.calc_train_acc(trues, predicts)
        return val_acc, total_loss

    def test(self):
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.eval()
        result = {}
        with torch.no_grad():
            for i, (texts, labels) in enumerate(self.dataset.test_iter):
                outputs = self.model(texts)
                labels = labels.cpu().numpy()
                predicts = outputs.cpu().numpy()
                for label, predict in zip(labels, predicts):
                    result[label] = predict
        return result

    def calc_train_acc(self, trues, predicts):
        length = len(trues)
        tot = 0
        for true, predict in zip(trues, predicts):
            tot += predict[true]
        return tot / length
