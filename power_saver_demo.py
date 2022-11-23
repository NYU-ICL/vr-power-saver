#!/usr/bin/env python
from pathlib import Path

from imageio import imread, imwrite
import numpy as np

from invoker import Script
from util.vr_tools import build_ecc_map, build_transition_mask


class PowerSaverDemo(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {
            "data_path": "./io/image_data",
            "image_fn": "teaser_raw.jpg",
            "fov": 60,  # in degrees
            "transition_width": 3,  # in degrees
        }

    @classmethod
    def modules(cls):
        return {
            # Add module dependencies
            "color_model": "base",
        }

    @classmethod
    def build_config(cls, args):
        # Args post-processing prior to script main exec
        args.update({
            "path": "./io/color_model",
            "model_fn": "model.pth",
            "power_weights": [231.5384684, 245.6795914, 530.7596369, 977.2813229],
        })
        return args

    def run(self):
        # Load Image
        data_root = Path(self.opt.data_path)
        img = imread(data_root / self.opt.image_fn, pilmode="RGB") / 255.

        # Resize Input Image
        def square_crop(img):
            hpad = (img.shape[1] - img.shape[0]) // 2
            out = img[:, hpad:hpad+img.shape[0], ...].copy()
            return out
        inp = square_crop(img)

        # Initialize Power Function Gradient
        power_vec = -np.array(self.opt.power_weights[:-1])

        # Load Model
        model_root = Path(self.opt.path)
        self.color_model.load(model_root / self.opt.model_fn)

        # Compute Model Output
        ecc_map = build_ecc_map(self.opt.fov, 0., 0.,
            self.color_model.opt.max_eccentricity, inp.shape[0], inp.shape[1])
        out = self.color_model.apply_filter(inp, ecc_map, power_vec)

        # Blend Model Output
        mask = build_transition_mask(ecc_map,
            self.color_model.opt.min_eccentricity, self.opt.transition_width)
        out = inp * (1 - mask) + out * mask

        imwrite("io/figures/teaser_gt.png", inp)
        imwrite("io/figures/teaser_op.png", out)


if __name__ == "__main__":
    PowerSaverDemo().initialize().run()
