import logging

import torch
from torch import nn
from torch.optim import Adam

from configs.arguments import TrainingArguments


# def accuracy_loss(outputs, labels):
#     # outputs: [b, 2]
#     # labels: [b]
#     result = 0
#     return torch.Tensor([0])


class Trainer:
    def __init__(self, config: TrainingArguments, model: nn.Module, dataset, time: str):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.time = time
        # loss: (b, 2) (b) -> num
        self.loss = nn.CrossEntropyLoss(reduction="none")
        if config.loss != 'CrossEntropy':
            # if config.loss == 'Accuracy':
            #     self.loss = accuracy_loss
            assert False
        self.save_path = './saved_dict/' + self.config.model_name + self.time + '.ckpt'

    def train(self):
        train_iter, val_iter = self.dataset.train_iter, self.dataset.val_iter
        # set state
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        best_val_loss = float('inf')
        last_improve = 0  # last time to update best_val_loss
        current_batch = 1

        for epoch in range(self.config.num_epoches):
            train_iter.shuffle()
            trues, predicts = [], []
            # texts: [tensor[8, 512], tensor[8, 512], tensor[8, 512]]  x, seq_len, mask
            # labels: tensor[8]
            for i, (texts, labels) in enumerate(train_iter):
                # outputs: [8, 2]
                outputs = self.model(texts)
                self.model.zero_grad()
                loss = torch.sum(self.loss(outputs, labels))  # TODO: check this loss
                loss.backward()
                optimizer.step()
                logging.info(f"Training, {i}/{len(train_iter)}, {epoch}/{self.config.num_epoches}, loss: {loss.item()}")

                trues.append(labels.cpu())
                predicts.append(outputs.cpu())

                if current_batch % self.config.show_period == 0:
                    # output accuracy
                    train_acc = self.calc_train_acc(trues, predicts)
                    val_acc, val_loss = self.eval(val_iter)
                    logging.info(f"Ep {epoch}/{self.config.num_epoches}, iter {current_batch},"
                                 f" train loss {loss.item()}, train acc {train_acc},"
                                 f" val loss {val_loss}, val acc {val_acc}, last upd {last_improve}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), self.save_path)
                        last_improve = current_batch
                        logging.info("Good, saving model.")
                    self.model.train()
                current_batch += 1
                if current_batch - last_improve > self.config.early_stop_diff:
                    logging.info("No improve for long, early stopped.")
                    return

    def eval(self, val_iter):
        self.model.eval()
        total_loss = 0
        trues, predicts = [], []
        logging.info("Evaluating...")
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
        self.model.load_state_dict(torch.load(self.save_path))
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
        length = len(trues) * self.config.batch_size
        tot = 0
        for true, predict in zip(trues, predicts):
            assert len(true) == len(predict) == self.config.batch_size
            for i in range(len(true)):
                if predict[i][true[i]] > predict[i][1 - true[i]]:
                    tot += 1
        return tot / length
