# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Combine 3 images to produce properly-scaled RGB images with arbitrary scaling.

The three images must be aligned and have the same pixel scale and size.
"""

from collections.abc import Sequence
from warnings import catch_warnings, filterwarnings

import numpy as np
from astropy.visualization import BaseInterval, BaseStretch, LinearStretch, ManualInterval

_OUTPUT_IMAGE_FORMATS = [float, np.float64, np.uint8]

__all__ = ["make_rgb", "RGBImageMapping"]


class RGBImageMapping:
    """
    Class to map red, green and blue images into either a normalized float or
    an 8-bit image.

    Optional clipping and application of a scaling function is performed to
    each band independently.

    Parameters
    ----------
    interval : `~astropy.visualization.BaseInterval` | Sequence, optional
        The interval object to apply to the data, either a single instance or
        a sequence for R, G, B.
    stretch : `~astropy.visualization.BaseStretch`, optional
        The stretch object to apply to the data.
    """

    def __init__(
        self,
        interval: BaseInterval | Sequence[BaseInterval] = ManualInterval(vmin=0, vmax=None),
        stretch: BaseStretch = LinearStretch(),
    ):
        intervals = 3 * [interval] if not isinstance(interval, Sequence) else interval
        if len(intervals) != 3:
            raise ValueError("Please provide 1 or 3 intervals.")

        self.intervals = intervals
        self.stretch = stretch

    def make_rgb_image(
        self, image_r: np.ndarray, image_g: np.ndarray, image_b: np.ndarray, output_dtype: type = np.uint8
    ):
        """
        Convert 3 arrays (image_r, image_g, and image_b) into an RGB image,
        either as an 8-bit per-channel or normalized image.

        The input images can be int or float, and in any range or bit-depth,
        but must have the same shape (NxM).

        Parameters
        ----------
        image_r : `numpy.ndarray`
            Image to map to red.
        image_g : `numpy.ndarray`
            Image to map to green.
        image_b : `numpy.ndarray`
            Image to map to blue.
        output_dtype : `numpy.dtype`, optional
            Image output format.

        Returns
        -------
        RGBimage : `numpy.ndarray`
            RGB color image with the specified format as an NxMx3 numpy array.
        """
        if output_dtype not in _OUTPUT_IMAGE_FORMATS:
            raise ValueError(f"'output_dtype' must be one of {_OUTPUT_IMAGE_FORMATS}!")

        image_r = np.asarray(image_r)
        image_g = np.asarray(image_g)
        image_b = np.asarray(image_b)

        if (image_r.shape != image_g.shape) or (image_g.shape != image_b.shape):
            msg = "The image shapes must match. r: {}, g: {} b: {}"
            raise ValueError(msg.format(image_r.shape, image_g.shape, image_b.shape))

        image_rgb = self.apply_mappings(image_r, image_g, image_b)
        if np.issubdtype(output_dtype, float):
            conv_images = self._convert_images_to_float(image_rgb, output_dtype)
        elif np.issubdtype(output_dtype, np.unsignedinteger):
            conv_images = self._convert_images_to_uint(image_rgb, output_dtype)

        return np.moveaxis(conv_images, 0, -1)

    def apply_mappings(self, image_r: np.ndarray, image_g: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        """
        Apply mapping intervals and stretch to convert image_r, image_g,
        and image_b images to a triplet of normalized images.

        Parameters
        ----------
        image_r : `numpy.ndarray`
            Image to be mapped to red
        image_g : `numpy.ndarray`
            Image to be mapped to green.
        image_b : `numpy.ndarray`
            Image to be mapped to blue.

        Returns
        -------
        image_rgb : `numpy.ndarray`
            Triplet of mapped images based on the specified (per-band)
            intervals and the stretch function.
        """
        image_rgbs = [image_r, image_g, image_b]
        for i, image in enumerate(image_rgbs):
            # Using syntax from mpl_normalize.ImageNormalize,
            # but NOT using that class to avoid dependency on matplotlib.

            # Define vmin and vmax
            vmin, vmax = self.intervals[i].get_limits(image)

            # copy because of in-place operations after
            image = np.array(image, copy=True, dtype=float)

            # Normalize based on vmin and vmax
            np.subtract(image, vmin, out=image)
            np.true_divide(image, vmax - vmin, out=image)

            # Clip to the 0 to 1 range
            np.clip(image, 0.0, 1.0, out=image)

            # Stretch values
            self.stretch(image, clip=False, out=image)

            image_rgbs[i] = image

        return np.asarray(image_rgbs)

    def _convert_images_to_float(self, image_rgb: np.ndarray, output_dtype: type):
        """Convert a triplet of normalized images to float."""
        return image_rgb.astype(output_dtype)

    def _convert_images_to_uint(self, image_rgb: np.ndarray, output_dtype: type):
        """Convert a triplet of normalized images to unsigned integers."""
        pixmax = float(np.iinfo(output_dtype).max)
        image_rgb *= pixmax
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
            return image_rgb.astype(output_dtype)


def make_rgb(
    image_r: np.ndarray,
    image_g: np.ndarray,
    image_b: np.ndarray,
    interval: BaseInterval | Sequence[BaseInterval] = ManualInterval(vmin=0, vmax=None),
    stretch: BaseStretch = LinearStretch(),
    filename: str | None = None,
    output_dtype: type = np.uint8,
):
    """
    Return a Red/Green/Blue color image from 3 images using
    a specified stretch and interval, for each band *independently*.

    The input images can be int or float, and in any range or bit-depth,
    but must have the same shape (NxM).

    For a more detailed look at the use of this method, see the document
    :ref:`astropy:astropy-visualization-rgb`.

    Parameters
    ----------
    image_r : `numpy.ndarray`
        Image to map to red.
    image_g : `numpy.ndarray`
        Image to map to green.
    image_b : `numpy.ndarray`
        Image to map to blue.
    interval : `~astropy.visualization.BaseInterval` | `Sequence`, optional
        The interval object to apply to the data; either a single instance or
        an array for R, G, B.
    stretch : `~astropy.visualization.BaseStretch`, optional
        The stretch object to apply to the data.
    filename : str, optional
        Write the resulting RGB image to a file (type determined by extension).
    output_dtype : `numpy.dtype`, optional
        Image output data type.

    Returns
    -------
    rgb : `numpy.ndarray`
        RGB color image, either float or integer with 8-bits per channel,
        as an NxMx3 numpy array.

    Notes
    -----
    This procedure of clipping and then scaling is similar to the DS9
    image algorithm (see the DS9 reference guide:
    http://ds9.si.edu/doc/ref/how.html).

    """
    map_ = RGBImageMapping(interval=interval, stretch=stretch)
    rgb = map_.make_rgb_image(image_r, image_g, image_b, output_dtype=output_dtype)

    if filename:
        import matplotlib.image

        matplotlib.image.imsave(filename, rgb, origin="lower")

    return rgb
