

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 56 / 30
hist_days = 7
fct_days = 1


class ElectricityLoadDataset(Dataset):
    """Sample data from electricity load dataset (per household, resampled to one hour)."""

    def __init__(self, df, samples, hist_len=24 * hist_days, fct_len=24 * fct_days):
        self.hist_num = hist_len
        self.fct_num = fct_len
        self.hist_len = pd.Timedelta(hours=hist_len)
        self.fct_len = pd.Timedelta(hours=fct_len)
        self.offset = pd.Timedelta(hours=1)
        self.samples = samples

        self.max_ts = df.index.max() - self.hist_len - self.fct_len + self.offset
        self.raw_data = df.copy()

        assert samples <= self.raw_data[:self.max_ts].shape[0]

        self.sample()

    def sample(self):
        """Sample individual series as needed."""

        # Calculate actual start for each household
        self.clean_start_ts = (self.raw_data != 0).idxmax()

        households = []

        for hh in self.raw_data.columns:
            hh_start = self.clean_start_ts[hh]
            hh_nsamples = min(self.samples, self.raw_data.loc[hh_start:self.max_ts].shape[0])

            hh_samples = (self.raw_data
                          .loc[hh_start:self.max_ts]
                          .index
                          .to_series()
                          .sample(hh_nsamples, replace=False)
                          .index)
            households.extend([(hh, start_ts) for start_ts in hh_samples])

        self.samples = pd.DataFrame(households, columns=("household", "start_ts"))

        self.raw_data = self.raw_data / self.raw_data[self.raw_data != 0].mean() - 1


        # Adding calendar features
        self.raw_data["yearly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofyear / 366)
        self.raw_data["weekly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofweek / 7)
        self.raw_data["daily_cycle"] = np.sin(2 * np.pi * self.raw_data.index.hour / 24)
        self.raw_data["weekend"] = (self.raw_data.index.dayofweek < 5).astype(float)
        self.raw_data["night"] = ((self.raw_data.index.hour < 7) & (self.raw_data.index.hour>21)).astype(float)
        self.raw_data["winter"] = ((self.raw_data.index.month < 4) & (self.raw_data.index.month > 10)).astype(float)
        self.raw_data["Holiday"] = ((self.raw_data.index.is_year_end | self.raw_data.index.is_year_start) | (( self.raw_data.index.month==12) & (self.raw_data.index.daysinmonth==25))).astype(float)
        self.raw_data["winter_daily_cycle"] =self.raw_data["daily_cycle"]*self.raw_data["night"]
        self.raw_data["summer_daily_cycle"] =self.raw_data["daily_cycle"]*(1-self.raw_data["night"] )

        #self.calendar_features = ["yearly_cycle", "weekly_cycle", "daily_cycle"]
        self.calendar_features = ["yearly_cycle", "weekly_cycle", "daily_cycle", "weekend", "night", "winter", "Holiday", "summer_daily_cycle", "winter_daily_cycle"]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        household, start_ts = self.samples.iloc[idx]

        hs, he = start_ts, start_ts + self.hist_len - self.offset
        fs, fe = he + self.offset, he + self.fct_len

        hist_data = self.raw_data.loc[hs:, [household] + self.calendar_features].iloc[:self.hist_num]
        fct_data = self.raw_data.loc[fs:, [household] + self.calendar_features].iloc[:self.fct_num]
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        return (torch.Tensor(hist_data.values).to(device),
                torch.Tensor(fct_data.values).to(device))