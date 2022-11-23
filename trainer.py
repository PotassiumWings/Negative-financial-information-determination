import logging

import torch
from torch import nn
from torch.optim import Adam

from configs.arguments import TrainingArguments
from models.loss import Loss


class Trainer:
    def __init__(self, config: TrainingArguments, model: nn.Module, dataset, time: str):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.time = time
        self.loss_config = Loss(config)
        self.loss = self.loss_config.get_loss()
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
            # text: [tensor[8, 512], tensor[8, 512], tensor[8, 512]]  x, seq_len, mask
            # entity: [tensor[8, 512], tensor[8, 512], tensor[8, 512]]  x, seq_len, mask
            # label: tensor[8], 0/1 y
            for i, (text, entity, label) in enumerate(train_iter):
                # outputs: [8]  or [8, 2] if cross entropy
                outputs = self.model(text, entity)
                self.model.zero_grad()

                loss = self.loss(outputs, label)
                loss.backward()
                optimizer.step()
                logging.info(f"Training, {i}/{len(train_iter)}, {epoch}/{self.config.num_epoches}, "
                             f"loss: {round(loss.item(), 4)}")

                trues.append(label.cpu())
                predicts.append(outputs.cpu())

                if current_batch % self.config.show_period == 0:
                    # output accuracy
                    train_acc = self.calc_train_acc(trues, predicts)
                    val_acc, val_loss = self.eval(val_iter)
                    logging.info(f"Ep {epoch}/{self.config.num_epoches}, iter {current_batch},"
                                 f" train loss {round(loss.item(), 4)}, train acc {round(train_acc, 4)},"
                                 f" val loss {round(val_loss, 4)}, val acc {round(val_acc, 4)},"
                                 f" last upd {last_improve}")
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
        # total_loss = 0
        trues, predicts = [], []
        logging.info("Evaluating...")
        with torch.no_grad():
            for i, (text, entity, label) in enumerate(val_iter):
                outputs = self.model(text, entity)
                trues.append(label.cpu())
                predicts.append(outputs.cpu())

        val_acc = self.calc_train_acc(trues, predicts)
        val_loss = self.loss(torch.cat(predicts), torch.cat(trues)).item()
        return val_acc, val_loss

    def test(self, filename):
        if filename == "":
            filename = self.save_path
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        result = {}
        with torch.no_grad():
            for i, (text, entity, label) in enumerate(self.dataset.test_iter):
                outputs = self.model(text, entity)
                label = label.cpu().numpy()
                predicts = outputs.cpu().numpy()
                for sub_label, predict in zip(label, predicts):
                    result_id, entity_name = sub_label.split(";")
                    if sub_label not in result:
                        result[result_id] = []
                    if predict < self.loss_config.gap:
                        continue
                    result[result_id].append(entity_name)
        return result

    def calc_train_acc(self, trues, predicts):
        length = len(trues) * self.config.batch_size
        tot = 0

        for true, predict in zip(trues, predicts):
            assert len(true) == len(predict) == self.config.batch_size
            for i in range(len(true)):
                if self.config.loss == "CrossEntropyLoss":
                    if predict[i][true[i]] > 0.5:
                        tot += 1
                else:
                    if true[i] == 1 and predict[i] > self.loss_config.gap or \
                            true[i] == 0 and predict[i] < self.loss_config.gap:
                        tot += 1
        return tot / length
