from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from invoker import Module
from util.colorspace import sRGB2RGB, RGB2sRGB, RGB2XYZ, XYZ2LMS, LMS2iDKL
import util.torch_rbf as rbf


class BaseColorModel(Module):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {
            "max_lm_contrast": 0.3025,
            "max_s_contrast": 0.00655,
            "min_eccentricity": 10,  # deg
            "max_eccentricity": 35,  # deg
            "layer_widths": [3, 2],  # (C_LM, C_S, e) --> (a, b)
            "layer_centres": [5],
            "rng_seed": 0,
        }

    def initialize(self):
        super(BaseColorModel, self).initialize()
        torch.manual_seed(self.opt.rng_seed)
        self.model = Network(
            self.opt.layer_widths, self.opt.layer_centres, rbf.gaussian,
            self.opt.max_lm_contrast, self.opt.max_s_contrast, self.opt.max_eccentricity)

    def optimize(self, x, y, nepochs, batch_size, lr, loss_func):
        self.model.train()
        obs = x.size(0)
        train_set = SimpleDataset(x, y)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        epoch = 0
        for epoch in range(nepochs):
            current_loss = 0
            progress = 0
            for batch_id, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_hat = self.model.forward(x_batch)
                loss = loss_func(y_hat, y_batch)
                current_loss += 1/(batch_id+1) * (loss.item() - current_loss)
                loss.backward()
                optimizer.step()
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch %d, Progress: %d/%d, Loss %f\t' %\
                                 (epoch+1, progress, obs, current_loss*10000))

    def eval(self, x):
        self.model.eval()
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        return self.model.forward(x)
    
    def compute_ellipses(self, img, ecc_map):
        # Convert Image to iDKL
        RGB2iDKL = LMS2iDKL @ XYZ2LMS @ RGB2XYZ
        img_idkl = (RGB2iDKL @ sRGB2RGB(img.reshape(-1, 3)).T).T

        # Compute Neutral Gray for each pixel
        lum = img_idkl[:, -1]
        pedestal = np.stack([lum, lum, lum])
        pedestal_dkl = (RGB2iDKL @ pedestal).T[..., :-1]

        # Compute DKL coodinates
        img_dkl = img_idkl[..., :-1] / (pedestal_dkl + 1e-9) - 1

        # Evaluate Model
        ecc_map = ecc_map.reshape(-1, 1)
        inp = np.concatenate((img_dkl, ecc_map), axis=-1)
        ellipse_wh_dkl = self.eval(inp).detach().numpy()

        # Convert Output to iDKL
        ellipse_wh_idkl = abs(pedestal_dkl * ellipse_wh_dkl)
        return ellipse_wh_idkl

    def apply_filter(self, img, ecc_map, energy_vec):
        RGB2iDKL = LMS2iDKL @ XYZ2LMS @ RGB2XYZ
        iDKL2RGB = np.linalg.inv(RGB2iDKL)

        # Compute Energy Normal Vector in iDKL
        energy_vec_idkl = energy_vec @ iDKL2RGB

        # Remove Luminance Dimension
        energy_vec_idkl = energy_vec_idkl[:-1]

        # Evaluate Pixel Color Deltas in iDKL
        ellipses_wh_idkl = self.compute_ellipses(img, ecc_map)
        denom = np.sqrt(
            np.sum(ellipses_wh_idkl**2 * energy_vec_idkl**2, axis=-1, keepdims=True))
        img_idkl = (RGB2iDKL @ sRGB2RGB(img.reshape(-1, 3)).T).T
        delta_idkl = ellipses_wh_idkl**2 * energy_vec_idkl / (denom + 1e-9)

        # Apply Color Shift
        out_idkl = img_idkl.copy()
        out_idkl[:, :-1] += delta_idkl

        # Convert Image to sRGB
        out = (iDKL2RGB @ out_idkl.T).T.reshape(img.shape)
        return RGB2sRGB(out)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def dump_weights(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for k, v in self.model.state_dict().items():
            df = pd.DataFrame(v.numpy())
            df.to_csv(root / k, index=False, header=False)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)


class Network(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func, max_lm, max_s, max_ecc):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.output_layer = nn.Sigmoid()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))

        # Hardcode maximum supported contrast
        w = torch.tensor([1 / max_lm, 1 / max_s, 1 / max_ecc], dtype=torch.float32)
        b = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.normalize = lambda x: w * x + b
        self.rescale_output = lambda x: x * torch.tensor([max_lm, max_s])

    def forward(self, x):
        out = self.normalize(x)
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return self.rescale_output(self.output_layer(out))
