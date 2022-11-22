import torch
from transformers import BertTokenizer, BertModel


class BasicConfig(object):
    def __init__(self):
        self.dataPath = "./train"  # 数据所在地址
        self.model_name = "bert"  # BERT模型所在地址
        self.tokenizer = BertTokenizer.from_pretrained("./" + self.model_name)
        self.model = BertModel.from_pretrained("./" + self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_file = "./data/train.csv"
        self.test_file = "./data/test.csv"
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'

        # bert constraints: max length is 512
        self.max_seq_len = 512
        self.f_max_seq_len = 129
        self.t_max_seq_len = self.max_seq_len - self.f_max_seq_len

        self.batch_size = 8

        # train : valid
        self.train_percent = 0.8

        self.learning_rate = 5e-5
        self.num_epoches = 10
        self.show_period = 10

        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768

        # more than 100 batch no improve, stop
        self.early_stop_diff = 100
