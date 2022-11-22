import pydantic_argparse
from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    model_name: str = Field("bert-base-chinese")

    train_file: str = Field("./data/train.csv")
    test_file: str = Field("./data/test.csv")

    max_seq_len: int = Field(512)
    f_max_seq_len: int = Field(129)

    batch_size: int = Field(8)

    # train : all
    train_percent: float = Field(0.9)

    learning_rate: float = Field(5e-5)
    num_epoches: int = Field(10)
    show_period: int = Field(10)

    hidden_dropout_prob: float = Field(0.1)
    hidden_size: int = Field(768)

    # more than ? batch no improve, stop
    early_stop_diff: int = Field(500)

    add_special_tokens: bool = Field(False)

    seed: int = Field(0)

    loss: str = Field("BCELoss")
    label_smoothing: float = Field(0.001)


# Create Parser and Parse Args
parser = pydantic_argparse.ArgumentParser(
    model=TrainingArguments,
    prog="python app.py",
    description="Training model job.",
    version="0.0.1",
    epilog="Training model job.",
)

args = parser.parse_typed_args()
