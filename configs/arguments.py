import pydantic_argparse
from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    model_name: str = Field("bert-base-chinese")
    mask_str: str = Field("[MASK]")

    data_path: str = Field("./data/")
    data_prefix: str = Field("")
    train_file: str = Field("train.csv")
    test_file: str = Field("test.csv")

    max_seq_len: int = Field(512)
    f_max_seq_len: int = Field(129)

    # train : all
    train_percent: float = Field(0.8)

    learning_rate: float = Field(5e-5)
    num_epoches: int = Field(10)
    show_period: int = Field(80)

    hidden_dropout_prob: float = Field(0.1)
    hidden_size: int = Field(768)

    # more than ? batch no improve, stop
    early_stop_diff: int = Field(1000)

    add_special_tokens: bool = Field(False)

    seed: int = Field(0)
    batch_size: int = Field(8)
    accumulate: int = Field(8)
    loss: str = Field("BCEWithLogitsLoss")
    label_smoothing: float = Field(0.001)

    prompt_positive: str = Field("好赞")
    prompt_negative: str = Field("差坏")
    prompt: bool = Field(False)
    prompt_pattern: int = Field(0)
    prompt_loss: str = Field("sum")

    model_filename: str = Field("")
    fine_tune: bool = Field(False)
    replace_entity: bool = Field(True)


# Create Parser and Parse Args
parser = pydantic_argparse.ArgumentParser(
    model=TrainingArguments,
    prog="python app.py",
    description="Training model job.",
    version="0.0.1",
    epilog="Training model job.",
)

args = parser.parse_typed_args()
