#%%
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime


# %%
data_path = "/Users/wonhyung64/data/trajectory/traj.csv"
df = pd.read_csv(data_path)

datetime.timestamp(datetime.strptime(tmp, "%Y-%m-%d %H:%M:%S+09:00"))
n = df.shape[0]

input = df.iloc[:, ]
target = df.iloc[:, -1].values.reshape(n, 1)

df[df["id"] == "traj_0"][["datetime", "lon", "lat"]].values