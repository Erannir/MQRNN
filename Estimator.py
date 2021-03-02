import pathlib
import pandas as pd
import numpy as np

from ElectricityLoadDataset import ElectricityLoadDataset
from MQRNN import MQRNN
import MQRNN

DATA_DIR = pathlib.Path("data")

if __name__ == "__main__":
    eldata = pd.read_parquet(DATA_DIR.joinpath("LD2011_2014.parquet"))
    eldata = eldata.resample("1H", on="timestamp").mean()

    ds = ElectricityLoadDataset(eldata, 100)
    data = 3

    input_size = 1  # y
    embed_size = 3  # x
    hidden_size = 64  # for lstm
    context_size = 8  # for c_t/c_a
    horizon = 24
    quantiles = [0.1, 0.5, 0.9]
    mq = MQRNN(input_size, embed_size, hidden_size, context_size, horizon, quantiles)


