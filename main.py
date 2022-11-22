import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from configs.arguments import TrainingArguments
from dataset.processor import Dataset
from models.basic_model import BasicModel
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

    setup_seed(config.seed)

    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicModel(config).to(device)

    logging.info("Loading dataset...")
    dataset = Dataset(config)
    trainer = Trainer(config, model, dataset)

    logging.info("Start Training.")
    trainer.train()
    # result = trainer.test()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
