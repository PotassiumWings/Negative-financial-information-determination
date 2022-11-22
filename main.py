from models.basic_model import BasicModel
from configs.arguments import TrainingArguments
from dataset.processor import Dataset
from trainer import Trainer
from datetime import datetime
import torch
import logging


def main(config: TrainingArguments):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logger = logging.getLogger()
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time = datetime.strftime(datetime.now(), "%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/log{time}.txt')
    file_handler.setFormatter(formats)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicModel(config).to(device)
    logging.info("Loading dataset...")
    dataset = Dataset(config)
    trainer = Trainer(config, model, dataset)

    logging.info("Start Training.")
    trainer.train()
    # result = trainer.test()
