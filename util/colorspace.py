#!/usr/bin/env python
import numpy as np

# xyY Color Space: https://en.wikipedia.org/wiki/CIE_1931_color_space
def XYZ2xyY(XYZ):
    xyz = XYZ / np.sum(XYZ)
    return xyz[0], xyz[1], XYZ[1]

def xyY2XYZ(x, y, Y):
    return np.array([Y/y*x, Y, Y/y*(1-x-y)])

def computeRGB2jvXYZMatrix():
    # sRGB definition: https://en.wikipedia.org/wiki/SRGB
    oldRGB2XYZ = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])

    # Judd-Vos correction: https://onlinelibrary.wiley.com/doi/epdf/10.1002/col.5080030309    
    def jvCorrect(x, y, Y):
        denom = 0.03845 * x + 0.01496 * y + 1
        xp = (1.0271 * x - 0.00008 * y - 0.00009) / denom
        yp = (0.00376 * x + 1.0072 * y + 0.00764) / denom
        return xp, yp, Y

    RGB2XYZjv = np.array([
        xyY2XYZ(*jvCorrect(*XYZ2xyY(oldRGB2XYZ @ np.array([1,0,0])))),
        xyY2XYZ(*jvCorrect(*XYZ2xyY(oldRGB2XYZ @ np.array([0,1,0])))),
        xyY2XYZ(*jvCorrect(*XYZ2xyY(oldRGB2XYZ @ np.array([0,0,1]))))
    ]).T

    return RGB2XYZjv

# DKL Color Space: https://www.cvrl.org/gallery/DKL-space.htm
LMS2iDKL = np.array([
    [1, -1, 0],
    [-1, -1, 1],
    [1, 1, 0],
])
iDKL2LMS = np.linalg.inv(LMS2iDKL)

# jvXYZ and LMS Conversion: http://www.cvrl.org/database/text/cones/sp.htm
XYZ2LMS = np.array([
    [0.15514, 0.54312, -0.03286],
    [-0.15514, 0.45684, 0.03286],
    [0, 0, 0.00801],
])
LMS2XYZ = np.linalg.inv(XYZ2LMS)

# jvXYZ and RGB Conversion: See computeRGB2SYZjv matrix
RGB2XYZ = computeRGB2jvXYZMatrix()
XYZ2RGB = np.linalg.inv(RGB2XYZ)

# sRGB definition: https://en.wikipedia.org/wiki/SRGB
def sRGB2RGB(sRGB):
    sRGB = np.clip(sRGB, 0, 1)
    lo_mask = sRGB <= 0.04045
    out = np.zeros_like(sRGB)
    out[lo_mask] = sRGB[lo_mask] / 12.92
    out[~lo_mask] = ((sRGB[~lo_mask] + 0.055) / 1.055) ** 2.4
    return out

def RGB2sRGB(RGB):
    RGB = np.clip(RGB, 0, 1)
    lo_mask = RGB<= 0.0031308
    out = np.zeros_like(RGB)
    out[lo_mask] = 12.92 * RGB[lo_mask]
    out[~lo_mask] = 1.055 * RGB[~lo_mask] ** (1/2.4) - 0.055
    return out

def generate_color_triangle(lum_srgb):
    lum = sRGB2RGB(lum_srgb)
    RGB2iDKL = LMS2iDKL @ XYZ2LMS @ RGB2XYZ
    iDKL2RGB = np.linalg.inv(RGB2iDKL)
    lm_prim = RGB2iDKL[0] / RGB2iDKL[2] * lum
    s_prim = RGB2iDKL[1] / RGB2iDKL[2] * lum
    lm_min, lm_max = lm_prim.min(), lm_prim.max()
    s_min, s_max = s_prim.min(), s_prim.max()
    lm = np.linspace(lm_min, lm_max, 1000)
    s = np.linspace(s_min, s_max, 1000)
    lmlm, ss = np.meshgrid(lm, s)

    def get_ab(p1, p2):
        return np.linalg.inv(np.array([[p1[0], 1], [p2[0], 1]])) @ np.array([p1[1], p2[1]])
    ab1 = get_ab([lm_prim[0], s_prim[0]], [lm_prim[1], s_prim[1]])
    ab2 = get_ab([lm_prim[1], s_prim[1]], [lm_prim[2], s_prim[2]])
    ab3 = get_ab([lm_prim[2], s_prim[2]], [lm_prim[0], s_prim[0]])
    mask1 = (lmlm * ab1[0] + ab1[1]) < ss
    mask2 = (lmlm * ab2[0] + ab2[1]) < ss
    mask3 = (lmlm * ab3[0] + ab3[1]) > ss
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(mask1 & mask2 & mask3)

    lum_grid = np.ones_like(lmlm) * lum
    dkl_grid = np.stack([lmlm, ss, lum_grid])
    rgb_grid = (iDKL2RGB @ dkl_grid.reshape(3, -1)).reshape(dkl_grid.shape).transpose(1, 2, 0)
    r_mask = ((rgb_grid[..., 0] > 0) & (rgb_grid[..., 0] < 1))
    g_mask = ((rgb_grid[..., 1] > 0) & (rgb_grid[..., 1] < 1))
    b_mask = ((rgb_grid[..., 2] > 0) & (rgb_grid[..., 2] < 1))
    mask = r_mask & g_mask & b_mask
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(mask)
    #plt.show()
    srgb_grid = RGB2sRGB(rgb_grid * mask[..., None])
    return srgb_grid, mask

def calibrate_iDKL_colorspace(srgb):
    dkl_pedestal = LMS2iDKL @ XYZ2LMS @ RGB2XYZ @ sRGB2RGB(srgb)

    def get_line(XYZ1, XYZ2):
        xyY1 = XYZ2xyY(XYZ1)
        xyY2 = XYZ2xyY(XYZ2)
        a, b = np.polyfit([xyY1[0], xyY2[0]], [xyY1[1], xyY2[1]], 1)
        return a, b

    def find_intersect(a1, b1, a2, b2):
        A = np.array([
            [a1, -1],
            [a2, -1],
        ])
        b = np.array([-b1, -b2])
        return np.linalg.inv(A) @ b

    def contrast(val, ped):
        return (val - ped) / np.abs(ped)

    def max_contrast(dir):
        r = RGB2XYZ[:, 0]
        g = RGB2XYZ[:, 1]
        b = RGB2XYZ[:, 2]

        origin = RGB2XYZ @ sRGB2RGB(srgb)
        point = LMS2XYZ @ iDKL2LMS @ (dkl_pedestal + dir)
        alpha, beta = get_line(origin, point)

        def intersect_colorpair_extreme(c1, c2):
            pair_alpha, pair_beta = get_line(c1, c2)
            int_xy = find_intersect(pair_alpha, pair_beta, alpha, beta)
            int_dkl = LMS2iDKL @ XYZ2LMS @ xyY2XYZ(*int_xy, origin[1])
            return np.abs(contrast(int_dkl @ dir, dkl_pedestal @ dir))

        return np.min([
            intersect_colorpair_extreme(r, b),
            intersect_colorpair_extreme(b, g),
            intersect_colorpair_extreme(r, g),
        ])
        
    lm_contrast = max_contrast(np.array([1, 0, 0]))
    s_contrast = max_contrast(np.array([0, 1, 0]))

    return dkl_pedestal, lm_contrast, s_contrast

def generate_dkl_grid(pedestal, width):
    # Calibrate iDKL color space
    dkl_pedestal, lm_contrast, s_contrast = calibrate_iDKL_colorspace(pedestal)

    centres = np.array([
        [dkl_pedestal[0] * (1 - lm_contrast * 0.5), dkl_pedestal[1], dkl_pedestal[2]], # 12 o'clock
        [dkl_pedestal[0], dkl_pedestal[1] * (1 + 0.5 * s_contrast), dkl_pedestal[2]], # 9 o'clock
        [dkl_pedestal[0], dkl_pedestal[1], dkl_pedestal[2]], # center
        [dkl_pedestal[0], dkl_pedestal[1] * (1 - 0.5 * s_contrast), dkl_pedestal[2]], # 3 o'clock
        [dkl_pedestal[0] * (1 + lm_contrast * 0.5), dkl_pedestal[1], dkl_pedestal[2]], # 12 o'clock
    ])

    # Setup the iDKL color grid
    LM = np.linspace(
        dkl_pedestal[0] * (1 - lm_contrast),
        dkl_pedestal[0] * (1 + lm_contrast),
        width,
    )
    S = np.linspace(
        dkl_pedestal[1] * (1 + s_contrast),
        dkl_pedestal[1] * (1 - s_contrast),
        width,
    )
    LMLM, SS = np.meshgrid(LM, S)
    VV = np.ones_like(LMLM) * dkl_pedestal[2]
    dkl = np.stack([LMLM, SS, VV], axis=-1)

    # Convert iDKL2sRGB
    dkl_flat = dkl.transpose(2, 0, 1).reshape(3, width**2)
    srgb = RGB2sRGB(XYZ2RGB @ LMS2XYZ @ iDKL2LMS @ dkl_flat)
    srgb = srgb.reshape(3, width, width).transpose(1, 2, 0)

    return dkl, srgb, centres
