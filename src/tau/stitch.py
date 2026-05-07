"""
Utilities for introspecting astronomy data.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from lsst.afw.image import ExposureF, Mask
from lsst.geom import Box2I, Point2I
from lsst.images import GeneralizedImage

__all__ = [
    "stitch_exposures",
    "stitch_generalized_images",
]


def stitch_exposures(
    images: Sequence[ExposureF],
    overlap: str = "overwrite",
    ramp_width: int = 512,
):
    """Stitch aligned ``ExposureF`` objects into a single larger exposure.

    Parameters
    ----------
    images : `~collections.abc.Sequence` [`~lsst.afw.image.ExposureF`]
        Input exposures to place into one output parent frame.
        All inputs are assumed to already share the same pixel grid;
        this function does not warp between WCS solutions.
    overlap : `str`, optional
        Policy for pixels covered by more than one exposure.

        - ``"overwrite"``: later exposures replace earlier ones.
        - ``"average"``: overlapping pixels are combined with equal weights.
        - ``"ramp"``: overlapping pixels are feathered using edge-distance
          weights that rise from 0 to 1 over ``ramp_width`` pixels.
    ramp_width : `int`, optional
        Width of the feathering zone used when ``overlap="ramp"``.

    Returns
    -------
    stitched : `lsst.afw.image.ExposureF`
        Output exposure spanning the union of the input bounding boxes.

    Notes
    -----
    Image values are combined according to ``overlap``.
    Masks are bitwise-ORed over contributing pixels.
    Variance is propagated with the same weights used for the image,
    using ``sum(w_i^2 var_i) / sum(w_i)^2``.
    """
    images = list(images)
    if not images:
        raise ValueError("image sequence must contain at least one ExposureF")

    valid_modes = {"overwrite", "average", "ramp"}
    if overlap not in valid_modes:
        raise ValueError(f"overlap must be one of {sorted(valid_modes)}, got {overlap!r}")
    if ramp_width <= 0:
        raise ValueError("ramp_width must be positive")

    min_x = min(im.getBBox().getMinX() for im in images)
    min_y = min(im.getBBox().getMinY() for im in images)
    max_x = max(im.getBBox().getMaxX() for im in images)
    max_y = max(im.getBBox().getMaxY() for im in images)
    out_bbox = Box2I(Point2I(min_x, min_y), Point2I(max_x, max_y))

    first = images[0]
    stitched = ExposureF(out_bbox, first.getWcs())
    stitched.mask.conformMaskPlanes(first.mask.getMaskPlaneDict())
    no_data = Mask.getPlaneBitMask("NO_DATA")
    stitched.maskedImage.set(np.nan, no_data, np.inf)

    try:
        stitched.setFilter(first.getFilter())
    except Exception:
        pass
    try:
        stitched.setPhotoCalib(first.getPhotoCalib())
    except Exception:
        pass

    if overlap == "overwrite":
        for im in images:
            stitched.maskedImage.assign(im.maskedImage, im.getBBox())
        return stitched

    out_image = stitched.image.array
    out_mask = stitched.mask.array
    out_var = stitched.variance.array
    sum_w = np.zeros_like(out_image, dtype=np.float32)  # weight per pixel
    sum_i = np.zeros_like(out_image, dtype=np.float32)  # weighted image sum
    sum_v = np.zeros_like(out_var, dtype=np.float32)  # weighted variance sum

    def _bbox_slices(bbox):
        y0 = bbox.getMinY() - out_bbox.getMinY()
        y1 = bbox.getMaxY() - out_bbox.getMinY() + 1
        x0 = bbox.getMinX() - out_bbox.getMinX()
        x1 = bbox.getMaxX() - out_bbox.getMinX() + 1
        return slice(y0, y1), slice(x0, x1)

    def _ramp_weights(shape):
        height, width = shape
        xdist = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1)).astype(np.float32)
        ydist = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1)).astype(np.float32)
        wx = np.clip(xdist / ramp_width, 0.0, 1.0)
        wy = np.clip(ydist / ramp_width, 0.0, 1.0)
        return wy[:, None] * wx[None, :]

    for im in images:
        bbox = im.getBBox()
        ys, xs = _bbox_slices(bbox)
        image = im.image.array.astype(np.float32, copy=False)
        mask = im.mask.array
        variance = im.variance.array.astype(np.float32, copy=False)

        if overlap == "average":
            weights = np.ones_like(image, dtype=np.float32)
        else:
            weights = _ramp_weights(image.shape)

        sum_w[ys, xs] += weights
        sum_i[ys, xs] += weights * image
        sum_v[ys, xs] += (weights**2) * variance
        out_mask[ys, xs] |= mask

    valid = sum_w > 0
    out_image[valid] = sum_i[valid] / sum_w[valid]
    out_var[valid] = sum_v[valid] / (sum_w[valid] ** 2)
    out_mask[valid] = np.bitwise_and(out_mask[valid], np.bitwise_not(no_data).astype(out_mask.dtype))
    return stitched


def stitch_generalized_images(
    images: Sequence[GeneralizedImage],
    overlap: str = "overwrite",
    ramp_width: int = 512,
):
    """Stitch aligned ``lsst.images.GeneralizedImage`` objects.

    Parameters
    ----------
    images : `~collections.abc.Sequence` [`~lsst.images.GeneralizedImage`]
        Input images derived from ``lsst.images.GeneralizedImage``. All inputs
        are assumed to already share the same pixel grid; this function does
        not warp between projections.
    overlap : `str`, optional
        Policy for pixels covered by more than one image.

        - ``"overwrite"``: later images replace earlier ones.
        - ``"average"``: overlapping pixels are combined with equal weights.
        - ``"ramp"``: overlapping pixels are feathered using edge-distance
          weights that rise from 0 to 1 over ``ramp_width`` pixels.
    ramp_width : `int`, optional
        Width of the feathering zone used when ``overlap="ramp"``.

    Returns
    -------
    stitched
        A stitched object of the same class as the first input.
    """
    images = list(images)
    if not images:
        raise ValueError("image sequence must contain at least one image")

    valid_modes = {"overwrite", "average", "ramp"}
    if overlap not in valid_modes:
        raise ValueError(f"overlap must be one of {sorted(valid_modes)}, got {overlap!r}")
    if ramp_width <= 0:
        raise ValueError("ramp_width must be positive")

    if not all(isinstance(im, GeneralizedImage) for im in images):
        raise TypeError("all images must derive from lsst.images.GeneralizedImage")

    def _get_image_array(image):
        if hasattr(image, "image"):
            return image.image.array
        if hasattr(image, "array"):
            return image.array
        raise TypeError(f"{type(image).__name__} must provide image.array or array")

    min_x = min(im.bbox.x.start for im in images)
    min_y = min(im.bbox.y.start for im in images)
    max_x = max(im.bbox.x.stop for im in images)  # exclusive
    max_y = max(im.bbox.y.stop for im in images)  # exclusive
    out_width = max_x - min_x
    out_height = max_y - min_y

    first = images[0]
    out_bbox = type(first.bbox).from_shape(
        shape=(out_height, out_width),
        start=(min_y, min_x),
    )
    first_image = _get_image_array(first)
    image_dtype = first_image.dtype
    trailing_shape = first_image.shape[2:]

    has_mask = hasattr(first, "mask") and getattr(first, "mask") is not None
    has_variance = hasattr(first, "variance") and getattr(first, "variance") is not None

    if has_mask:
        mask_dtype = first.mask.array.dtype
    if has_variance:
        var_dtype = first.variance.array.dtype

    out_shape = (out_height, out_width, *trailing_shape)
    out_image = np.zeros(out_shape, dtype=image_dtype)
    out_mask = np.zeros(out_shape[:2], dtype=mask_dtype) if has_mask else None
    out_var = np.full(out_shape[:2], np.inf, dtype=var_dtype) if has_variance else None

    if overlap == "overwrite":
        for im in images:
            y0 = im.bbox.y.start - min_y
            y1 = im.bbox.y.stop - min_y
            x0 = im.bbox.x.start - min_x
            x1 = im.bbox.x.stop - min_x
            out_image[y0:y1, x0:x1] = _get_image_array(im)
            if has_mask:
                assert out_mask is not None
                out_mask[y0:y1, x0:x1] = im.mask.array
            if has_variance:
                assert out_var is not None
                out_var[y0:y1, x0:x1] = im.variance.array
    else:
        sum_w = np.zeros(out_shape[:2], dtype=np.float32)
        sum_i = np.zeros(out_shape, dtype=np.float32)
        sum_v = np.zeros(out_shape[:2], dtype=np.float32) if has_variance else None

        def _ramp_weights(shape):
            height, width = shape
            xdist = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1)).astype(np.float32)
            ydist = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1)).astype(np.float32)
            wx = np.clip(xdist / ramp_width, 0.0, 1.0)
            wy = np.clip(ydist / ramp_width, 0.0, 1.0)
            return wy[:, None] * wx[None, :]

        for im in images:
            y0 = im.bbox.y.start - min_y
            y1 = im.bbox.y.stop - min_y
            x0 = im.bbox.x.start - min_x
            x1 = im.bbox.x.stop - min_x

            image = _get_image_array(im).astype(np.float32, copy=False)
            variance = im.variance.array.astype(np.float32, copy=False) if has_variance else None

            if overlap == "average":
                weights = np.ones(image.shape[:2], dtype=np.float32)
            else:
                weights = _ramp_weights(image.shape[:2])

            sum_w[y0:y1, x0:x1] += weights

            weights_view = weights if image.ndim == 2 else weights[:, :, None]
            sum_i[y0:y1, x0:x1] += weights_view * image

            if has_variance:
                assert sum_v is not None
                assert variance is not None
                sum_v[y0:y1, x0:x1] += (weights**2) * variance
            if has_mask:
                assert out_mask is not None
                out_mask[y0:y1, x0:x1] |= im.mask.array

        valid = sum_w > 0
        out_image[valid] = sum_i[valid] / (sum_w[valid][:, None] if out_image.ndim == 3 else sum_w[valid])
        if has_variance:
            assert out_var is not None
            assert sum_v is not None
            out_var[valid] = sum_v[valid] / (sum_w[valid] ** 2)

    projection = getattr(first, "projection", None)
    image_cls = cast(Any, type(first))

    # Try direct ndarray-based constructor first (e.g., ColorImage).
    try:
        if projection is not None:
            return image_cls(out_image, bbox=out_bbox, projection=projection)
        return image_cls(out_image, bbox=out_bbox)
    except Exception:
        pass

    # Fallback to plane-based constructor (e.g., MaskedImage-like classes).
    if not hasattr(first, "image"):
        raise TypeError(f"Could not construct stitched {type(first).__name__} from array/bbox constructor")
    image_plane = type(first.image)(out_image, bbox=out_bbox, projection=projection)
    kwargs = {"image": image_plane, "projection": projection}
    if has_mask:
        kwargs["mask"] = type(first.mask)(out_mask, bbox=out_bbox, schema=getattr(first.mask, "schema", None))
    if has_variance:
        kwargs["variance"] = type(first.variance)(out_var, bbox=out_bbox, projection=projection)
    if has_mask and hasattr(first.mask, "schema"):
        kwargs["mask_schema"] = first.mask.schema
    return image_cls(**kwargs)
