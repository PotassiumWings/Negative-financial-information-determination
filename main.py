from models.basic_model import BasicModel
from configs.basic_config_bert import BasicConfig
from dataset.processor import Dataset
from trainer import Trainer


if __name__ == '__main__':
    config = BasicConfig()
    model = BasicModel(config).to(config.device)
    dataset = Dataset(config)
    trainer = Trainer(config, model, dataset)

    trainer.train()
    # result = trainer.test()
