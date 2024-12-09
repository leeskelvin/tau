"""
Tools for image manipulation and visualization.
"""

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization import (
    AsinhStretch,
    AsymmetricPercentileInterval,
    BaseInterval,
    BaseStretch,
    HistEqStretch,
    ImageNormalize,
    LinearStretch,
    LogStretch,
    MinMaxInterval,
    PercentileInterval,
    PowerDistStretch,
    PowerStretch,
    SinhStretch,
    ZScaleInterval,
)

__all__ = ["aimage", "colorbar"]


def _parse_inputs(
    image: Any, mask: Any = None, mask_plane_dict: dict | None = None
) -> tuple[np.ndarray, np.ndarray | None, dict | None]:
    """Parse an image to a numpy array.

    Parameters
    ----------
    image : Any
        The image data. This can be a numpy array, an `Exposure`, a
        `MaskedImage`, or an `Image`. If mask and mask plane dictionary
        information are not explicitly provided, they are extracted from
        `image` if available.
    mask : Any, optional
        The mask data, if available.
    mask_plane_dict : dict | None, optional
        The mask plane dictionary, if available.

    Returns
    -------
    image : np.ndarray
        The image data as a numpy array.
    mask : np.ndarray | None
        The mask data as a numpy array, if available.
    mask_plane_dict : dict | None
        The mask plane dictionary, if available.
    """
    # mask_plane_dict
    if mask_plane_dict is None and hasattr(image, "mask"):
        mask_plane_dict = image.mask.getMaskPlaneDict()
    elif mask_plane_dict is None and hasattr(mask, "getMaskPlaneDict"):
        mask_plane_dict = mask.getMaskPlaneDict()
    # mask
    if mask is None and hasattr(image, "mask"):
        mask = image.mask.array
    # image
    if hasattr(image, "image"):
        image = image.image.array
    elif hasattr(image, "getImage"):
        image = image.getImage().array
    return image, mask, mask_plane_dict


def _get_vmin_vmax(
    image: np.ndarray, interval: str | BaseInterval, pc: int | float | Sequence[int | float], contrast: float
):
    """Get vmin and vmax values for an image based on the interval.

    Parameters
    ----------
    image : np.ndarray
        The image data.
    interval : str | BaseInterval
        The interval to use for vmin and vmax calculation.
    pc : int | float | Sequence[int | float]
        The percentile(s) to use for the interval.
    contrast : float
        The contrast to use for the ZScale interval.

    Returns
    -------
    vmin, vmax : int, float
        The vmin and vmax values.
    """
    pc = [pc] if isinstance(pc, (int, float)) else pc
    match interval:
        case "percentile":
            if (len(pc) == 2 and pc == [0, 100]) or (len(pc) == 1 and pc[0] == 100):
                interval = MinMaxInterval()
            else:
                interval = AsymmetricPercentileInterval(*pc) if len(pc) == 2 else PercentileInterval(pc[0])
        case "zscale":
            interval = ZScaleInterval(contrast=contrast)
        case _:
            if not isinstance(interval, BaseInterval):
                raise ValueError("interval must be a known string or a BaseInterval instance.")
    vmin, vmax = interval.get_limits(image)
    return vmin, vmax


def _get_stretch(
    stretch_str: str,
    a: float | None,
    slope: float = 1.0,
    intercept: float = 0.0,
    image: np.ndarray | None = None,
    vmin: int | float | None = None,
    vmax: int | float | None = None,
):
    """Get a stretch object based on the stretch string.

    Parameters
    ----------
    stretch_str : str
        The stretch to use, as a named string.
    a : float | None
        The index parameter passed to the stretch. The meaning of this
        varies by stretch. If None, the stretch class default is used.
    slope : float
        The slope parameter for `~astropy.visualization.LinearStretch`.
    intercept : float
        The intercept parameter for `~astropy.visualization.LinearStretch`.
    image : np.ndarray | None
        The image data for `~astropy.visualization.HistEqStretch`.
    vmin : int | float | None
        The minimum value for `~astropy.visualization.HistEqStretch`.
    vmax : int | float | None
        The maximum value for `~astropy.visualization.HistEqStretch`.
    """
    args = {"a": a} if a is not None else {}
    match stretch_str:
        case "linear":
            stretch = LinearStretch(slope, intercept)  # type: ignore
        case "power":
            stretch = PowerStretch(**args)
        case "powerDist":
            stretch = PowerDistStretch(**args)
        case "log":
            stretch = LogStretch(**args)
        case "asinh":
            stretch = AsinhStretch(**args)
        case "sinh":
            stretch = SinhStretch(**args)
        case "histEq":
            assert image is not None
            stretch = HistEqStretch(np.clip(image, vmin, vmax))
        case _:
            raise ValueError(f"Unknown stretch: {stretch_str}")
    return stretch


def colorbar(mappable: ScalarMappable) -> Colorbar:
    """Create a colorbar for a given mappable.

    Parameters
    ----------
    mappable : ScalarMappable
        The mappable object to create a colorbar for.

    Returns
    -------
    cbar : Colorbar
        The colorbar object.
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def aimage(
    image: Any,
    mask: Any = None,
    mask_plane_dict: dict | None = None,
    interval: str | BaseInterval = "percentile",
    stretch: str | BaseStretch = "linear",
    vmin: None | int | float = None,
    vmax: None | int | float = None,
    pc: int | float | Sequence[int | float] = 100,
    contrast: float = 0.25,
    a: float = 2.0,
    slope: float = 1.0,
    intercept: float = 0.0,
    cmap="grey",
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 300,
    show_cbar: bool | None = None,
    show_mask: bool = False,
):
    image, mask, mask_plane_dict = _parse_inputs(image, mask, mask_plane_dict)
    assert isinstance(image, np.ndarray)
    assert mask is None or isinstance(mask, np.ndarray)
    assert mask_plane_dict is None or isinstance(mask_plane_dict, dict)

    if vmin is None and vmax is None:
        vmin, vmax = _get_vmin_vmax(image, interval, pc, contrast)
    assert vmin is not None and vmax is not None

    if isinstance(stretch, str):
        stretch = _get_stretch(stretch, a, slope, intercept, image, vmin, vmax)
    assert isinstance(stretch, BaseStretch)

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)  # type: ignore
    assert isinstance(norm, Normalize)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image, origin="lower", norm=norm, cmap=cmap)

    if show_mask:
        pass

    if show_cbar is None:
        show_cbar = True if image.ndim == 2 else False

    if show_cbar and norm is not None:
        cbar = colorbar(im)
        cbar.set_ticks(norm.inverse(np.linspace(0, 1, 11)))

    plt.show()
