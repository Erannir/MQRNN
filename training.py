import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

import torch
from torch import optim

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ElectricityLoadDataset import ElectricityLoadDataset
from loss_functions import quantile_loss
from Models import MQRNN

random_seed = 42
torch.manual_seed(0)

path = pathlib.Path.cwd()
DATA_DIR = pathlib.Path("data")
MODEL_DIR = pathlib.Path("models")
GRAPH_DIR = pathlib.Path("graphs")


def retrieve_data(samples=100, val_split=0.2, test_split=0.1, features=range(9), batch_size=32, workers=8, dir="data"):
    # eldata = pd.read_parquet(DATA_DIR.joinpath("LD2011_2014.txt"))
    df = pd.read_csv(pathlib.Path(dir).joinpath("LD2011_2014.txt"),
                         parse_dates=[0],
                         delimiter=";",
                         decimal=",")
    df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
    df = df.resample("1H", on="timestamp").mean()

    ds = ElectricityLoadDataset(df, samples, features=features)

    ds_size = len(ds)
    val_size = int(val_split * ds_size)
    test_size = int(test_split * ds_size)
    split_sizes = [ds_size - val_size - test_size, val_size, test_size]
    train_ds, val_ds, test_ds = random_split(ds, split_sizes)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    return train_loader, val_loader, test_loader


class Estimator:

    def __init__(self, batch_size, quantiles=[0.01, 0.25, 0.5, 0.75, 0.99]):
        self.batch_size = batch_size
        self.quantiles = quantiles

    def calculate_loss(self, model, enc_data, dec_data, input_size=1):
        y_e, x_e = enc_data[..., 0:input_size], enc_data[..., input_size:]
        y_d, x_d = dec_data[..., 0:input_size], dec_data[..., input_size:]

        predictions = model(y_e, x_e, x_d)
        loss = quantile_loss(predictions, y_d, self.quantiles)
        return loss

    def train(self, model, train_loader, val_loader, optimizer, scheduler=None, epochs=5, print_every=50):
        for epoch in range(epochs):
            print(str(datetime.datetime.now()) + ' Starting epoch %d / %d' % (epoch + 1, epochs))

            # Train Loop
            model.train()
            for t, (enc_data, dec_data) in enumerate(train_loader):
                enc_data, dec_data = enc_data, dec_data

                # (enc_data, dec_data) = ( (batch_size, seq_len, embed_size) , (batch_size, horizon, embed_size) )
                unified_data = torch.cat((enc_data, dec_data), 1)  # (batch_size, seq_len + horizon, embed_size)
                h = dec_data.shape[1]
                horizons = []
                for i in range(1, enc_data.shape[1]+1):
                    slice = unified_data[:, i:i+h, :]
                    horizons.append(slice)
                horizons = torch.stack(horizons, dim=1)  # (batch_size, seq_len, horizon, embed_size)

                loss = self.calculate_loss(model, enc_data, horizons)
                if (t + 1) % print_every == 0:
                    print(str(datetime.datetime.now()) + ' t = %d, loss = %.4f' % (t + 1, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Eval Loop
            model.eval()
            loss = 0
            for t, (enc_data, dec_data) in enumerate(val_loader):
                loss += self.calculate_loss(model, enc_data, dec_data)
            loss = loss / (t + 1)
            print(str(datetime.datetime.now()) + ' Got average loss of (%.2f)' % (loss), "last_lr =", optimizer.param_groups[0]['lr'])
            torch.save(model.state_dict(), MODEL_DIR.joinpath("epoch_{0}_loss_{1}".format(epoch, loss)))
            if scheduler:
                scheduler.step(loss)

    def forcast(self, model, enc_data, dec_data, quantiles=[0.01, 0.25, 0.5, 0.75, 0.99], input_size=1):
        model.eval()
        y_e, x_e = enc_data[..., 0:input_size], enc_data[..., input_size:]
        y_d, x_d = dec_data[..., 0:input_size], dec_data[..., input_size:]

        predictions = model(y_e, x_e, x_d)        # dimensions: (batch, len(hidden_states), horizon, len(quantiles))
        predictions = predictions.squeeze().detach().numpy()  # dimensions: (horizon, len(quantiles))

        len_hist = y_e.shape[1]
        len_fct = y_d.shape[1]
        forecast_range = np.arange(len_hist, len_hist + len_fct, 1)

        for i in range(y_e.shape[0]):
            y_h = y_e[i].squeeze()
            y_f = y_d[i].squeeze()
            plt.plot(np.arange(0, len_hist+1, 1), torch.cat((y_h, y_f[0].view(1))), label="historical data")

            plt.plot(forecast_range, y_f, label="actual", c="black")

            c = np.arange(len(quantiles)+1)
            norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            cmap.set_array([])
            for j, q in enumerate(quantiles[:-1]):
                plt.fill_between(forecast_range, predictions[i, :, j], predictions[i, :, j + 1], color=cmap.to_rgba(j+1))

            plt.legend(loc=0)
            plt.savefig(pathlib.Path(GRAPH_DIR).joinpath(str(i)))
            plt.close()


if __name__ == "__main__":
    input_size = 1  # y
    embed_size = 9  # x
    hidden_size = 30  # for lstm: "both with a state dimension of 30"
    context_size = 8  # for c_t/c_a
    horizon = 24
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    samples = 10
    batch_size = 32
    epochs = 1
    print_every = 50

    MODEL_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = retrieve_data(samples=samples)
    model = MQRNN(input_size, embed_size, hidden_size, context_size, horizon, quantiles)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)

    estimator = Estimator(batch_size=batch_size, quantiles=quantiles)
    print("Training info: samples={}, batch={}, printing training loss every {} batches".format(370 * samples,
                                                                                                batch_size,
                                                                                                print_every))
    estimator.train(model, train_loader, val_loader, optimizer, scheduler, epochs=epochs, print_every=print_every)

    model.eval()
    dataiter = iter(test_loader)
    enc_data, dec_data = dataiter.next()
    estimator.forcast(model, enc_data, dec_data, quantiles)
