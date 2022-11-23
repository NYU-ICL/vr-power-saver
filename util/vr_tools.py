import numpy as np

def build_ecc_map(fov, center_x, center_y, max_ecc, height, width):
    """
    Parameters
    ----------
    fov : float
        FoV to show the image at
    center_x: float
        x coordinate of center of gaze. normalized between [-1, 1]
    center_y: float
        y coordinate of center of gaze. normalized between [-1, 1]
    max_ecc: float
        maximum eccentricity to output for ecc_map
    height: int
        height of output array
    width: int
        with of output array
    """
    d = 1 / np.tan(fov * np.pi / 180 / 2)
    x = np.linspace(-1, 1, width) + center_x
    y = np.linspace(-1, 1, height) + center_y
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    ecc_map = np.arctan(dist / d)[..., None] * 180 / np.pi
    ecc_map[ecc_map > max_ecc] = max_ecc
    return ecc_map

def build_transition_mask(ecc_map, transition_ecc, transition_width):
    mask = ecc_map - (transition_ecc - transition_width / 2)
    mask[mask < 0] = 0
    mask[mask > transition_width] = transition_width
    mask /= transition_width
    return mask
