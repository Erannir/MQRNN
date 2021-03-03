import pathlib
import pandas as pd
import numpy as np

from ElectricityLoadDataset import ElectricityLoadDataset
from Models import MQRNN
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

DATA_DIR = pathlib.Path("data")

if __name__ == "__main__":
    eldata = pd.read_csv(DATA_DIR.joinpath("LD2011_2014.txt"),
                     parse_dates=[0],
                     delimiter=";",
                     decimal=",")
    eldata.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
    #eldata = pd.read_parquet(DATA_DIR.joinpath("LD2011_2014.txt"))
    eldata = eldata.resample("1H", on="timestamp").mean()

    dataset = ElectricityLoadDataset(eldata, 100)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

    input_size = 1  # y
    embed_size = 3  # x
    hidden_size = 30  # for lstm: "both with a state dimension of 30"
    context_size = 8  # for c_t/c_a
    horizon = 24
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    mq = MQRNN(input_size, embed_size, hidden_size, context_size, horizon, quantiles)

    # Run one step for debugging
    dataiter = iter(loader)
    enc_data, dec_data = dataiter.next()
    y_e, x_e = enc_data[:, :, 0].view(enc_data.shape[0], enc_data.shape[1], -1), enc_data[:, :, 1:]
    y_d, x_d = dec_data[:, :, 0].view(dec_data.shape[0], dec_data.shape[1], -1), dec_data[:, :, 1:]
    loss = mq(y_e, x_e, y_d, x_d)

    # Train
    optimizer = Adam(mq.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for enc_data, dec_data in enumerate(loader):
        y_e, x_e = enc_data[:, :, 0].view(enc_data.shape[0], enc_data.shape[1], -1), enc_data[:, :, 1:]
        y_d, x_d = dec_data[:, :, 0].view(dec_data.shape[0], dec_data.shape[1], -1), dec_data[:, :, 1:]
        loss = mq(y_e, x_e, y_d, x_d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
