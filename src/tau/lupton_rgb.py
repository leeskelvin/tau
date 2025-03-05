# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Combine 3 images to make a properly scaled RGB image a la Lupton et al. (2004).

The three images must be aligned and have the same pixel scale and size.

For details, see : https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L
"""

from collections.abc import Sequence
from warnings import catch_warnings, filterwarnings

import numpy as np
from astropy.visualization import BaseInterval, BaseStretch, ManualInterval, ZScaleInterval
from astropy.visualization.stretch import _prepare

from .basic_rgb import RGBImageMapping

__all__ = ["make_lupton_rgb", "LuptonAsinhStretch", "LuptonAsinhZscaleStretch", "RGBImageMappingLupton"]


def compute_intensity(
    image_r: np.ndarray, image_g: np.ndarray | None = None, image_b: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute average intensity from red, green and blue images.

    Parameters
    ----------
    image_r : `~numpy.ndarray`
        Image data mapped to the red channel, or total intensity if
        ``image_g`` and ``image_b`` are None.
    image_g : `~numpy.ndarray`, optional
        Image data mapped to the green channel.
    image_b : `~numpy.ndarray`, optional
        Image data mapped to the blue channel.

    Returns
    -------
    intensity : `~numpy.ndarray`
        Average intensity from all provided channels.
    """
    images = [image for image in (image_r, image_g, image_b) if image is not None]

    with catch_warnings():
        filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        intensity = np.nanmean(images, axis=0)
    return intensity


class LuptonAsinhStretch(BaseStretch):
    r"""
    A modified asinh stretch, with some changes to the constants
    relative to `~astropy.visualization.AsinhStretch`.

    The stretch is given by:

    .. math::
        & y = {\rm asinh}\left(\frac{soft * x}{span}\right) *
        \frac{scaling}{{\rm asinh}(scaling * soft)}

    Parameters
    ----------
    span : float, optional
        Linear data span of the image.
        The ``span`` parameter must be greater than 0.
    soft : float, optional
        The asinh softening parameter.
        The ``soft`` parameter must be greater than 0.
    scaling : float, optional
        The scaling fraction of the asinh stretch to apply.
        The ``scaling`` parameter must be greater than 0.

    Notes
    -----
    Based on the asinh stretch presented in Lupton et al. 2004
    (https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L).
    """

    def __init__(self, span: float = 5, soft: float = 8, scaling: float = 0.1):
        super().__init__()

        if span < 0:
            raise ValueError(f"The data span parameter must be non-negative! {span=}")
        if soft < 0:
            raise ValueError(f"The softening parameter must be non-negative! {soft=}")

        # 32bit floating point machine epsilon; sys.float_info.epsilon is 64bit
        epsilon = 1.0 / 2**23
        if abs(soft) < epsilon:
            soft = 0.1
        else:
            soft_max = 1e10
            if soft > soft_max:
                soft = soft_max

        self.span = span
        self.soft = soft
        self._slope = scaling / np.arcsinh(scaling * soft)
        self._softening = soft / float(span)

    def __call__(self, values: np.ndarray, clip: bool = False, out: np.ndarray | None = None):
        values = _prepare(values, clip=clip, out=out)
        np.multiply(values, self._softening, out=values)
        np.arcsinh(values, out=values)
        np.multiply(values, self._slope, out=values)
        return values


class LuptonAsinhZscaleStretch(LuptonAsinhStretch):
    r"""
    A modified asinh stretch, with the linear stretch calculated using ZScale.

    The stretch is given by:

    .. math::
        & y = {\rm asinh}\left(\frac{soft * x}{span}\right) *
        \frac{scaling}{{\rm asinh}(scaling * soft)} \\
        & span = z2 - z1

    Parameters
    ----------
    image : `~numpy.ndarray` | Sequence[`~numpy.ndarray`]
        The image to analyze, or a list of 3 images to be converted to an
        intensity image.
    soft : float, optional
        The asinh softening parameter.
        The ``soft`` parameger must be greater than 0.
    vmin : float | Sequence[float], optional
        The value, or array of values, to subtract from the images(s) before
        determining the zscaling.
    """

    def __init__(
        self,
        image: np.ndarray | Sequence[np.ndarray],
        soft: float = 8,
        vmin: float | Sequence[float] | None = None,
    ):
        # A copy, because of in-place operations after
        image = np.array(image, copy=True, dtype=float)

        _raiseerr = False
        if len(image.shape) == 2:
            image = [image]
        elif len(image.shape) == 3:
            if image.shape[0] != 3:
                _raiseerr = True
        else:
            _raiseerr = True

        if _raiseerr:
            raise ValueError(
                f"Input 'image' must be a single image or a 3xMxN array of 3 images! {image.shape=}"
            )

        image = list(image)  # needs to be mutable

        if vmin is not None:
            vmin = len(image) * [vmin] if not isinstance(vmin, Sequence) else vmin

            if len(vmin) != len(image):
                raise ValueError("vmin must have the same length as image.")
            for i, im in enumerate(image):
                if vmin[i] != 0.0:
                    image[i] = im - vmin[i]  # n.b. a copy

        image = compute_intensity(*image)
        zscale_limits = ZScaleInterval().get_limits(image)

        _span = zscale_limits[1] - zscale_limits[0]

        self._image = image

        super().__init__(span=_span, soft=soft)


class RGBImageMappingLupton(RGBImageMapping):
    """
    Class to map red, green and blue images into either a normalized float or
    an 8-bit image.

    RGB mapping is achieved by performing optional clipping and applying
    a scaling function to each band in non-independent manner that depends
    on the other bands, following the scaling scheme presented in
    Lupton et al. 2004.

    Parameters
    ----------
    interval : `~astropy.visualization.BaseInterval` | Sequence, optional
        The interval object to apply to the data, either a single instance or
        an array for R, G, B.
    stretch : `~astropy.visualization.BaseStretch`
        The stretch object to apply to the data.
    """

    def __init__(
        self,
        interval: BaseInterval | Sequence[BaseInterval] = ManualInterval(vmin=0, vmax=None),
        stretch: BaseStretch = LuptonAsinhStretch(span=5, soft=8, scaling=0.1),
    ):
        super().__init__(interval=interval, stretch=stretch)
        self._pixmax = 1.0

    def intensity(self, image_r, image_g, image_b):
        """
        Return the total intensity from the red, blue, and green intensities.
        This is a naive computation, and may be overridden by subclasses.

        Parameters
        ----------
        image_r : ndarray
            Intensity of image to be mapped to red; or total intensity if
            ``image_g`` and ``image_b`` are None.
        image_g : ndarray, optional
            Intensity of image to be mapped to green.
        image_b : ndarray, optional
            Intensity of image to be mapped to blue.

        Returns
        -------
        intensity : ndarray
            Total intensity from the red, green and blue intensities, or
            ``image_r`` if green and blue images are not provided.

        """
        return compute_intensity(image_r, image_g, image_b)

    def apply_mappings(self, image_r, image_g, image_b):
        """
        Apply mapping stretch and intervals to convert images image_r, image_g,
        and image_b to a triplet of normalized images, following the scaling
        scheme presented in Lupton et al. 2004.

        Compared to astropy's ImageNormalize, which first normalizes images
        by cropping and linearly mapping onto [0.,1.] and then applies
        a specified stretch algorithm, the Lupton et al. algorithm applies
        stretching to a multi-color intensity and then computes per-band
        scaled images with bound cropping.

        This is modified here by allowing for different minimum values
        for each of the input r, g, b images, and then computing
        the intensity on the subtracted images.

        Parameters
        ----------
        image_r : `~numpy.ndarray`
            Intensity of image to be mapped to red
        image_g : `~numpy.ndarray`
            Intensity of image to be mapped to green.
        image_b : `~numpy.ndarray`
            Intensity of image to be mapped to blue.

        Returns
        -------
        image_rgb : `~numpy.ndarray`
            Triplet of mapped images based on the specified (per-band)
            intervals and the stretch function.

        Notes
        -----
        The Lupton et al 2004 algorithm is computed with the following steps:

        1. Shift each band with the minimum values
        2. Compute the intensity I and stretched intensity f(I)
        3. Compute the ratio of the stretched intensity to intensity f(I)/I,
        and clip to a lower bound of 0
        4. Compute the scaled band images by multiplying with the ratio f(I)/I
        5. Clip each band to a lower bound of 0
        6. Scale down pixels where max(R,G,B)>1 by the value max(R,G,B)

        """
        image_r = np.array(image_r, copy=True)
        image_g = np.array(image_g, copy=True)
        image_b = np.array(image_b, copy=True)

        # Subtract per-band minima
        image_rgb = [image_r, image_g, image_b]
        for i, img in enumerate(image_rgb):
            vmin, _ = self.intervals[i].get_limits(img)
            image_rgb[i] = np.subtract(img, vmin)

        image_rgb = np.asarray(image_rgb)

        # Determine the intensity and streteched intensity
        intensity = self.intensity(*image_rgb)
        assert intensity is not None
        fI = self.stretch(intensity, clip=False)
        assert fI is not None

        # Get normalized fI, and clip to lower bound of 0:
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
            fInorm = np.where(intensity <= 0, 0, np.true_divide(fI, intensity))

        # Compute X = x * f(I) / I for each filter x=(r,g,b)
        np.multiply(image_rgb, fInorm, out=image_rgb)

        # Clip individual bands to minimum of 0, as
        # individual bands can be < 0 even if fI/I isn't.
        image_rgb = np.clip(image_rgb, 0.0, None)

        # Determine the max of all 3 bands at each position
        maxRGB = np.max(image_rgb, axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            image_rgb = np.where(
                maxRGB > self._pixmax,
                np.true_divide(image_rgb * self._pixmax, maxRGB),
                image_rgb,
            )

        return np.asarray(image_rgb)


def make_lupton_rgb(
    image_r: np.ndarray,
    image_g: np.ndarray,
    image_b: np.ndarray,
    interval: BaseInterval | Sequence[BaseInterval] | None = None,
    stretch: BaseStretch | str = "luptonAsinhZscale",
    span: float = 5,
    soft: float = 8,
    scaling: float = 0.1,
    filename: str | None = None,
    output_dtype: type = np.uint8,
):
    r"""
    Return an RGB color image from 3 images using interconnected band scaling,
    and an arbitrary stretch function (by default, an asinh stretch).

    The input images can be int or float, and in any range or bit-depth.

    For a more detailed look at the use of this method, see the document
    :ref:`astropy:astropy-visualization-rgb`.

    Parameters
    ----------
    image_r : ndarray
        Image to map to red.
    image_g : ndarray
        Image to map to green.
    image_b : ndarray
        Image to map to blue.
    interval : `~astropy.visualization.BaseInterval` | Sequence, optional
        The interval object to apply to the data, either a single instance or
        an array for R, G, B.
    stretch : `~astropy.visualization.BaseStretch`, optional
        The stretch object to apply to the data. If set, the input values of
        ``span``, ``soft`` and ``scaling`` are ignored. For the Lupton scheme,
        this may be either `~astropy.visualization.LuptonAsinhZscaleStretch`,
        `~astropy.visualization.LuptonAsinhStretch`, or another stretch.
    span : float, optional
        The linear stretch of the image.
    soft : float, optional
        The asinh softening parameter.
    scaling : float, optional
        The scaling fraction of the asinh stretch to apply.
    filename : str, optional
        Write the resulting RGB image to a file (type determined be extension).
    output_dtype : `numpy.dtype`, optional
        Image output data type.

    Returns
    -------
    rgb : `~numpy.ndarray`
        RGB color image as an NxMx3 array, in the specified data type format.
    """
    if interval is None:
        interval = ManualInterval(vmin=0, vmax=None)
    interval = interval if isinstance(interval, Sequence) else [interval] * 3
    assert [isinstance(ival, BaseInterval) for ival in interval]

    if isinstance(stretch, str):
        match stretch:
            case "luptonAsinh":
                stretch = LuptonAsinhStretch(span=span, soft=soft, scaling=scaling)
            case "luptonAsinhZscale":
                vmin = [ival.get_limits(img)[0] for ival, img in zip(interval, (image_r, image_g, image_b))]
                stretch = LuptonAsinhZscaleStretch(image=(image_r, image_g, image_b), soft=soft, vmin=vmin)
            case _:
                raise ValueError("Unknown stretch string.")
    if stretch is None:
        stretch = LuptonAsinhZscaleStretch(image=(image_r, image_g, image_b), soft=soft)
    assert isinstance(stretch, BaseStretch)

    lup_map = RGBImageMappingLupton(interval=interval, stretch=stretch)
    rgb = lup_map.make_rgb_image(image_r, image_g, image_b, output_dtype=output_dtype)

    if filename:
        import matplotlib.image

        matplotlib.image.imsave(filename, rgb, origin="lower")

    return rgb
