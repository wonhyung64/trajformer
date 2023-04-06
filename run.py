#%%
import os
import torch
import torch.nn as nn
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


sample = df[df["id"] == "traj_0"][["x", "y", "val", "acc", "dt", "dd", "label"]].values
x = torch.tensor(sample[..., :4], dtype=torch.float32).unsqueeze(0)
delta = torch.tensor(sample[..., 4:6], dtype=torch.float32).unsqueeze(0)
y = torch.tensor(sample[0, -1:], dtype=torch.float32).unsqueeze(0)


e = 4 # modify to 4 (lon, lat, vel, acc)
d = 64
L = 2
N_h = 4
r = 2
k = 9
b = 1
h = 16
t = 55

fc1 = nn.Linear(e, d)
st_mlp = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, h),
)
fc2 = nn.Linear(d, d)

delta[...,0]
a = torch.tensor([1, 2, 3])
i = torch.tensor([[0,1,0], [0,1,0], [0,1,0]])
a.unsqueeze(-1)*i
i

x1 = fc1(x).transpose(1, 2).unsqueeze(-1)

x2 = nn.functional.unfold(
    x1,
    kernel_size=(k, 1),
    dilation=1,
    padding=((k-1) // 2, 0),
    stride=1
    ).reshape(b, d, k, -1)
x3 = x2.permute(0, 3, 2, 1).reshape(-1, k, d)

fc2(x3).reshape(-1, k, h, d//h).shape
fc2(x3).reshape(110, 5)
x.shape


#%%


#%% delta
def compute_st_intervals(spatio_temporal_var, kernel_size):
    batch_size, traj_size = spatio_temporal_var.shape
    pad_size = kernel_size // 2
    padded_batch = nn.functional.pad(spatio_temporal_var, (pad_size, pad_size), mode='constant', value=-1e+5)

    indices = []
    for i in range(pad_size, pad_size+t):
        for j in range(i - pad_size, i + pad_size + 1):
            indices.append(j)

    indices_batch = torch.tensor(indices).tile(batch_size,).reshape(batch_size, -1)
    spatio_temporal_grid = torch.gather(input=padded_batch, dim=1, index=indices_batch).reshape(batch_size, traj_size, kernel_size)
    spatio_temporal_intervals = (spatio_temporal_grid - spatio_temporal_var.unsqueeze(-1))
    spatio_temporal_intervals = torch.where(abs(spatio_temporal_intervals) > 1e+5, 0., spatio_temporal_intervals)
    
    return spatio_temporal_intervals


sample = df[df["id"] == "traj_0"][["x", "y", "time"]]
x = torch.tensor(sample["x"].values).unsqueeze(0)
y = torch.tensor(sample["y"].values).unsqueeze(0)
time = torch.tensor(sample["time"].values).unsqueeze(0)

delta_dist = torch.sqrt(compute_st_intervals(x, 9)**2 + compute_st_intervals(y, 9)**2)
delta_time = compute_st_intervals(time, 9)

# %%
import torch.nn.utils.rnn as rnn_utils
rnn_utils.pad_packed_sequence(x, batch_first=True)

input_cols = ["x", "y", "val", "acc"]
delta_cols = ["x", "y", "time"]
target_cols = ["label"]
sample0 = df[df["id"] == "traj_0"][delta_cols].values
sample1 = df[df["id"] == "traj_1"][delta_cols].values

#TODO data loader with irregular length sequence

#%%
import matplotlib.pyplot as plt
traj = df[df["id"] ==  "traj_2"][["x", "y"]]
plt.scatter(traj["x"], traj["y"])
traj_scaled = 
df["x"].max()
df["x"].min()

scaled_y = np.round((traj["y"] - traj["y"].min())/(traj["y"].max()-traj["y"].min()) * 223).astype(int)
scaled_x = np.round((traj["x"] - traj["x"].min())/(traj["x"].max()-traj["x"].min()) * 223).astype(int)
index = np.vstack([scaled_x, scaled_y]).T


src = torch.ones(224, 224, 1, dtype=torch.float32)
torch.zeros(224, 224, 1, dtype=torch.float32).scatter(0, torch.tensor(index), src)
torch.zeros(224, 224, 1, dtype=torch.float32).scatter(0, torch.tensor(index), src)
torch.
plt.scatter(scaled_x, scaled_y)

x = torch.zeros(2,2,1)
index = torch.tensor([[0,1]])
x.scatter(0, index, torch.ones_like(x))

x = torch
torch.zeros(1,2,3).scatter(0, torch.tensor([[[0,0], [1,1]]]),1.)
torch.zeros(2,3).scatter(0, torch.tensor([[0], [1]]),1.)
torch.zeros(2).scatter(0, torch.tensor([0]),1.)

index
index.shape
a.shape
a = torch.LongTensor([[[2, 1], [2, 2], [3, 3]]])
K = 5
a = torch.LongTensor([index])
a.shape
out = torch.zeros(a.size(0), 224, 224)
out[torch.arange(out.size(0)).unsqueeze(1), a[:, :, 0], a[:, :, 1]] = 1.
print(out)
len(index)
torch.sum(out)


from torchvision.transforms.functional import to_pil_image
to_pil_image(out.view(224, 224))


ind= torch.tensor([[[1,3],
                   [0,3],
                   [1,2]],
                   ])
ind.shape
base = torch.ones(1, 3, 4)
base
base.scatter_(2, ind, 0)
base.shape

index.shape
src = torch.ones((2,5))
index = torch.tensor([[0,1,2,0,0]])
torch.zeros(3,5, dtype=src.dtype).scatter_add_(0, index, src)

index
index = np.unique(index, axis=0)
torch.zeros((224,224))[index[:,0], index[:,1]]

mask = hessian_mat.diagonal() == 0
        hess_tmp = hessian_mat + torch.diag(mask) * 1e-5

#%%