import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from configs.arguments import TrainingArguments
from dataset.processor import Dataset
from models.basic_model import BasicModel
from models.prompt_model import PromptModel
from trainer import Trainer


def main(config: TrainingArguments):
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("saved_dict"):
        os.mkdir("saved_dict")

    # logging config
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logger = logging.getLogger()
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time = datetime.strftime(datetime.now(), "%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/log{time}.txt')
    file_handler.setFormatter(formats)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    for x in config:
        logging.info(x)

    setup_seed(config.seed)

    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.prompt:
        model = PromptModel(config).to(device)
    else:
        model = BasicModel(config).to(device)

    logging.info("Loading dataset...")
    dataset = Dataset(config)
    trainer = Trainer(config, model, dataset, time)

    logging.info("Start Training.")

    if config.model_filename == "":
        trainer.train("AdamW", config.learning_rate)
        logging.info(f"Best val loss: {trainer.best_val_loss}")
    if config.fine_tune:
        logging.info(f"Fine-tuning with SGD and lr {config.learning_rate / 10}...")
        trainer.train("SGD", config.learning_rate / 10, config.model_filename)
        logging.info(f"Best val loss: {trainer.best_val_loss}")
    result = trainer.test(config.model_filename)
    generate_submission(result, dataset, time)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_submission(result, dataset, time):
    import csv
    with open(f"submission_{time}.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["id", "negative", "key_entity"])
        final_result = [[] for i in range(len(dataset.test_labels))]
        for row_index in result:
            row_label = dataset.index_to_label[row_index]
            row_index, entity = row_label.split(";")  # 0 小资钱包
            final_result[int(row_index)].append(entity)

        for row_index in range(len(final_result)):
            row_id = dataset.test_labels[row_index]  # 67bbbed4

            negative = 0
            if len(final_result[row_index]) > 0:
                negative = 1

            key_entity = ";".join(final_result[row_index])

            writer.writerow([row_id, negative, key_entity])
