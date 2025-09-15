"""
Tools for image manipulation and visualization.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
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
from astropy.wcs import WCS
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import Fixed
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

__all__ = ["colorbar", "aimage"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
logger.addHandler(handler)


def _parse_inputs(
    image: Any, mask: Any = None, mask_plane_dict: dict | None = None, wcs: Any | None = None
) -> tuple[np.ndarray, np.ndarray | None, dict | None, Any | None]:
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
    wcs : Any | None, optional
        The WCS information, if available.

    Returns
    -------
    image : np.ndarray
        The image data as a numpy array.
    mask : np.ndarray | None
        The mask data as a numpy array, if available.
    mask_plane_dict : dict | None
        The mask plane dictionary, if available.
    wcs : Any | None
        The WCS information in astropy format, if available.
    """
    # WCS
    if wcs is None:
        if hasattr(image, "wcs"):
            wcs = image.wcs
        elif hasattr(image, "getWcs"):
            wcs = image.getWcs()
    if wcs is not None and not isinstance(wcs, WCS):
        if hasattr(wcs, "getFitsMetadata"):
            try:
                wcs = WCS(wcs.getFitsMetadata().toDict())
            except RuntimeError:
                logger.warning("Failed to parse WCS from fits metadata.")
                wcs = None
        else:
            raise TypeError("WCS must be an instance of lsst.afw.geom.SkyWcs, astropy.wcs.WCS, or None.")
    # mask_plane_dict
    if mask_plane_dict is None and hasattr(image, "mask"):
        mask_plane_dict = image.mask.getMaskPlaneDict()
    elif mask_plane_dict is None and hasattr(mask, "getMaskPlaneDict"):
        mask_plane_dict = mask.getMaskPlaneDict()
    # mask
    if mask is None and hasattr(image, "mask"):
        mask = image.mask.array
    elif hasattr(mask, "image"):
        mask = mask.image.array
    elif hasattr(mask, "getImage"):
        mask = mask.getImage().array
    elif hasattr(mask, "getImageF"):
        mask = mask.getImageF().array
    elif hasattr(mask, "array"):
        mask = mask.array
    # image
    if hasattr(image, "image"):
        image = image.image.array
    elif hasattr(image, "getImage"):
        image = image.getImage().array
    elif hasattr(image, "getImageF"):
        image = image.getImageF().array
    elif hasattr(image, "array"):
        image = image.array

    if mask is not None:
        if image.shape != mask.shape:
            logger.warning("The image and mask shapes do not match.")

    return image, mask, mask_plane_dict, wcs


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
    pc = [pc] if isinstance(pc, int | float) else pc
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


def _add_mask(
    ax: Axes,
    mask: np.ndarray,
    mask_plane_dict: dict | None,
    extent: tuple[float, float, float, float],
    binsize: int,
    rot90: int,
    mask_planes: str | list[str] | None,
    mask_alpha: float,
    mask_fontsize: str | float,
    mask_loc: str | int,
    show_legend: bool,
):
    """Add a mask to an image plot.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to add the mask to.
    mask : np.ndarray
        The mask data.
    mask_plane_dict : dict | None
        The mask plane dictionary, associating the mask plane names (keys)
        with their binary bits (values).
    extent : tuple[float, float, float, float]
        The extent of the image in the format (xmin, xmax, ymin, ymax).
    binsize : int
        The bin size to apply to the mask.
    mask_planes : str | list[str] | None
        The mask planes to show. If None, all planes are shown.
    mask_alpha : float
        The alpha value for the mask overlay.
    mask_fontsize : str
        The fontsize for the mask legend.
    mask_loc : str
        The location for the mask legend.
    show_legend : bool
        Show the mask legend, if available.
    """
    if mask_plane_dict is None:
        mask_plane_bits = np.arange(int(np.log2(np.max(mask))) + 1)
        mask_plane_dict = {f"{mask_plane_bit}": mask_plane_bit for mask_plane_bit in mask_plane_bits}
    if mask_planes is None:
        mask_planes = list(mask_plane_dict.keys())
    if isinstance(mask_planes, str):
        mask_planes = [mask_planes]
    mask_plane_lookup = {
        bit: plane
        for plane, bit in sorted(mask_plane_dict.items(), key=lambda x: x[1])
        if plane in mask_planes
    }

    if binsize != 1:
        mask = block_reduce(mask, binsize, np.nanmean, 0, func_kwargs={"dtype": int})

    mask_bits = np.full_like(mask, np.nan, dtype=float)
    for bit, plane in reversed(mask_plane_lookup.items()):
        mask_bits[mask & 2**bit == 2**bit] = bit

    cmap = plt.get_cmap("tab20")
    if rot90 != 0:
        mask_bits = np.rot90(mask_bits, k=rot90)
    ax.imshow(
        mask_bits,
        cmap=cmap,
        alpha=mask_alpha,
        vmin=0,
        vmax=20,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        rasterized=True,
    )

    if show_legend:
        legend_patches = []
        for bit, plane in mask_plane_lookup.items():
            color = cmap(bit)
            legend_patches.append(Patch(color=color, label=plane))
        ax.legend(
            handles=legend_patches,
            loc=mask_loc,
            framealpha=0.5,
            borderaxespad=0.2,
            handlelength=0.9,
            columnspacing=0.9,
            handletextpad=0.5,
            fontsize=mask_fontsize,
            fancybox=False,
        )


def colorbar(
    mappable: ScalarMappable, norm: ImageNormalize | None = None, fixed_size: float = 0.15, pad: float = 0.05
) -> Colorbar:
    """Create a colorbar for a given mappable.

    Parameters
    ----------
    mappable : ScalarMappable
        The mappable object to create a colorbar for.
    norm : ImageNormalize | None, optional
        The normalization object to use for the colorbar.
    fixed_size : float, optional
        The fixed size of the colorbar (inches).
    pad : float, optional
        The padding between the mappable and the colorbar.

    Returns
    -------
    cbar : Colorbar
        The colorbar object.
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=Fixed(fixed_size), pad=pad, axes_class=Axes)
    breakpoint()
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)

    if norm is not None:
        cbar.set_ticks(norm.inverse(np.linspace(0, 1, 11)))

    # Generate unique tick labels with increasing precision as necessary
    tick_values = cbar.get_ticks()
    cbar.set_ticks(tick_values)
    cbar.ax.minorticks_off()
    for decimals in range(0, 7):
        formatted_tick_values = [f"{tick_value:.{decimals}f}" for tick_value in tick_values]
        if len(set(formatted_tick_values)) == len(tick_values):
            cbar.ax.set_yticklabels(formatted_tick_values)
            break

    return cbar


def aimage(
    # input data
    image: Any,
    mask: Any = None,
    mask_plane_dict: dict | None = None,
    wcs: Any | None = None,
    # image display
    xmin: None | int | float = None,
    xmax: None | int | float = None,
    ymin: None | int | float = None,
    ymax: None | int | float = None,
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
    fwhm: float = 0.0,
    binsize: int = 1,
    rot90: int = 0,
    # mask display
    mask_planes: str | list[str] | None = None,
    mask_alpha: float = 1.0,
    mask_fontsize: str | float = "xx-small",
    mask_loc: str | int = "upper left",
    # grid
    grid_color: str = "white",
    grid_linestyle: str = "solid",
    grid_linewidth: float = 1,
    # title
    title: object = None,
    title_fontsize: str | float = "x-small",
    title_loc: Literal["left", "center", "right"] = "left",
    # figure options
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 300,
    facecolor: str = "#ffffee",
    fig: Figure | None = None,
    ax: Axes | None = None,
    fname: str | None = None,
    # WCS
    decimal: bool = True,
    show_decimal_unit: bool = True,
    # show toggles
    show_axes: bool = True,
    show_minor_ticks: bool = False,
    show_cbar: bool | None = None,
    show_mask: bool = False,
    show_grid: bool = False,
    show_wcs: bool = False,
    show_legend: bool = True,
):
    """Display an image with optional mask overlay."""
    image, mask, mask_plane_dict, wcs = _parse_inputs(image, mask, mask_plane_dict, wcs)
    assert isinstance(image, np.ndarray)
    assert mask is None or isinstance(mask, np.ndarray)
    assert mask_plane_dict is None or isinstance(mask_plane_dict, dict)
    assert wcs is None or isinstance(wcs, WCS)

    xslice = slice(0 if xmin is None else int(xmin), image.shape[1] if xmax is None else int(xmax))
    yslice = slice(0 if ymin is None else int(ymin), image.shape[0] if ymax is None else int(ymax))
    xyslice = (yslice, xslice) if rot90 in [0, 2] else (xslice, yslice)
    image = image[xyslice]
    if mask is not None:
        mask = mask[xyslice]
    extent = (xyslice[1].start, xyslice[1].stop, xyslice[0].start, xyslice[0].stop)

    if vmin is None or vmax is None:
        vmin0, vmax0 = _get_vmin_vmax(image, interval, pc, contrast)
        vmin = vmin0 if vmin is None else vmin
        vmax = vmax0 if vmax is None else vmax
    assert vmin is not None and vmax is not None

    if isinstance(stretch, str):
        stretch = _get_stretch(stretch, a, slope, intercept, image, vmin, vmax)
    assert isinstance(stretch, BaseStretch)

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=False)  # type: ignore
    assert isinstance(norm, Normalize)

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, facecolor=facecolor, projection=wcs if show_wcs else None)
        external_fig = False
    else:
        if show_wcs:
            subplotspec = ax.get_subplotspec()
            if isinstance(subplotspec, SubplotSpec):
                fig.delaxes(ax)
                ax = fig.add_subplot(subplotspec, projection=wcs)
        external_fig = True

    if fwhm > 0:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        nan_pixels = np.isnan(image)
        image[nan_pixels] = 0
        image = gaussian_filter(image, sigma)
        image[nan_pixels] = np.nan

    if binsize != 1:
        image = block_reduce(image, binsize, np.nanmean, 0)

    if rot90 != 0:
        image = np.rot90(image, k=rot90)
    im = ax.imshow(
        np.clip(image, vmin, vmax),
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=extent,
        rasterized=True,
    )

    if show_grid:
        ax.grid(color=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth)

    try:
        ax.coords[0].set_format_unit("degree", decimal=decimal, show_decimal_unit=show_decimal_unit)
        ax.coords[1].set_format_unit("degree", decimal=decimal, show_decimal_unit=show_decimal_unit)
        ax.coords[0].set_ticks_position("bl")
        ax.coords[1].set_ticks_position("bl")
    except AttributeError:
        pass

    if show_minor_ticks:
        try:
            ax.coords[0].display_minor_ticks(True)
            ax.coords[1].display_minor_ticks(True)
        except AttributeError:
            ax.minorticks_on()

    if title is not None:
        ax.set_title(str(title), loc=title_loc, fontsize=title_fontsize)

    if show_mask and mask is not None:
        _add_mask(
            ax,
            mask,
            mask_plane_dict,
            extent,
            binsize,
            rot90,
            mask_planes,
            mask_alpha,
            mask_fontsize,
            mask_loc,
            show_legend,
        )

    if not show_axes:
        ax.set_axis_off()

    if show_cbar is None:
        show_cbar = True if image.ndim == 2 else False

    if show_cbar and norm is not None:
        _ = colorbar(im, norm=norm)

    if not external_fig:
        if fname is not None:
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    return {"vmin": vmin, "vmax": vmax, "stretch": stretch}
