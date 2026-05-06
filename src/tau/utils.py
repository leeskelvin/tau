"""
Utilities for introspecting astronomy data.
"""

import logging
import re
import sys
import textwrap
import time

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_skycoord
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureF, Mask
from lsst.daf.butler import DatasetRef
from lsst.geom import Box2I, Point2I
from lsst.pipe.base import Pipeline
from lsst.source.injection.utils._make_injection_pipeline import _parse_config_override
from lsst.utils.packages import getEnvironmentPackages

__all__ = [
    "print_session_info",
    "ref_to_title",
    "convert_selection_sets",
    "dump_config",
    "fit_lsst_wcs",
    "stitch_exposures",
]


def print_session_info():
    """Print session start time, Python interpreter, and LSST Science
    Pipelines package information.
    """
    # Time info
    print(f"# Session Info on {time.strftime('%Y-%m-%d at %H:%M:%S %Z', time.localtime(time.time()))}\n")

    # Python info
    print(f"## Python Interpreter\n\nVersion: {sys.version}  \nExecutable: {sys.executable}\n")

    # LSST info
    packages = getEnvironmentPackages(True)
    dev_packages = {"lsst_distrib": packages["lsst_distrib"]}
    dev_packages.update({k: v.split("@")[0] for k, v in packages.items() if "LOCAL" in v})
    print("## Science Pipelines\n\n" + "\n".join(f"{k:<20} {v}" for k, v in dev_packages.items()))


def ref_to_title(
    ref: DatasetRef,
    modifier: str = "",
    exclude: str | list[str] = "",
    delimiter: str = "\n",
    wrap=80,
    show_dataset_type: bool = True,
    show_data_id: bool = True,
    show_run: bool = True,
) -> str:
    """Convert a DatasetRef to a title string for a plot.

    Parameters
    ----------
    ref : `DatasetRef`
        The DatasetRef to convert.
    modifier : `str`, optional
        A modifier string to append to the title.
    exclude : `str` | `list`[`str`], optional
        Dimensions in the data ID to exclude.
    delimiter : `str`, optional
        The delimiter to use between the different parts of the title.

    Returns
    -------
    title : `str`
        The title string.
    """
    parts = []
    if show_dataset_type:
        parts.append(ref.datasetType.name + modifier)
    if show_data_id:
        data_id = {k: v for k, v in ref.dataId.required.items() if k not in exclude}
        data_id_str = ", ".join([f"{k}: {repr(v)}" for k, v in data_id.items()])
        parts.append(data_id_str)
    if show_run:
        parts.append(ref.run)
    wrapped_parts = [
        textwrap.fill(p, width=wrap, break_long_words=False, break_on_hyphens=False) for p in parts
    ]
    return delimiter.join(wrapped_parts)


def convert_selection_sets(obj: dict):
    """Convert all `SelectionSet` objects in a dictionary or list to lists.

    This is useful for dumping the config to YAML, as `SelectionSet` objects
    are not serializable by default. This function will recursively traverse
    the input object and convert any `SelectionSet` objects it finds to lists
    of their names.

    Parameters
    ----------
    obj : `dict`
        The object to convert. This can be a dictionary or a list that may
        contain `SelectionSet` objects.

    Notes
    -----
    This function modifies the input object in place. If you want to keep
    the original object unchanged, make a copy of it before calling this
    function.
    """
    for k, v in obj.items():
        if isinstance(v, dict):
            convert_selection_sets(v)
        elif hasattr(v, "__module__") and "configChoiceField" in v.__module__:
            # Check if the value is a SelectionSet and convert it to a list
            obj[k] = list(v)


def dump_config(
    pipeline: Pipeline | str,
    instrument: str | None = None,
    config: str | list[str] | None = None,
    log_level: int = logging.INFO,
):
    """Dump the config for an LSST pipeline YAML in YAML format.

    Parameters
    ----------
    pipeline : `~lsst.pipe.base.Pipeline` | `str`
        Location of a pipeline definition file in YAML format.
    instrument : `str`, optional
        Add instrument overrides. Must be a fully qualified class name.
    config : `str` | `list`[`str`], optional
        Config override for a task, in the format 'label:key=value'.
    log_level : `int`, optional
        The logging level to use.

    Returns
    -------
    pipeline_config_yaml : `str`
        The pipeline config in YAML format.
    """
    # Instantiate logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(handler)

    # Load the pipeline and apply config overrides, if supplied.
    if isinstance(pipeline, str):
        pipeline = Pipeline.fromFile(pipeline)
    else:
        pipeline = pipeline
    if config:
        if isinstance(config, str):
            config = [config]
        for conf in config:
            config_label, config_key, config_value = _parse_config_override(conf)
            pipeline.addConfigOverride(config_label, config_key, config_value)

    # Add an instrument override, if provided.
    if instrument:
        pipeline.addInstrument(instrument)

    pipeline_graph = pipeline.to_graph()

    # Loop over all tasks in the pipeline.
    pipeline_config = {}
    for task_node in pipeline_graph.tasks.values():
        task_label = task_node.label
        task_config = task_node.config.toDict()
        convert_selection_sets(task_config)
        pipeline_config[task_label] = task_config  # yaml.dump(task_config, sort_keys=False)

    pipeline_config_yaml = yaml.dump(pipeline_config, sort_keys=False, default_flow_style=None, width=110)

    # Amend all task labels to include the fully qualified task name
    for task_node in pipeline_graph.tasks.values():
        task_label = task_node.label
        task_class = task_node.task_class_name
        pattern = rf"^({re.escape(task_label)}:)"
        header = f"# {task_class}"
        bar = "# " + "=" * (len(task_class) + 0)
        replacement = f"\n{bar}\n{header}\n{bar}\n\\1"
        # replacement = f"\n\\1\n{bar}\n{header}\n{bar}"
        pipeline_config_yaml = re.sub(pattern, replacement, pipeline_config_yaml, flags=re.MULTILINE)

    return pipeline_config_yaml.strip()


def fit_lsst_wcs(
    wcs: SkyWcs,
    bbox: Box2I,
    num_points: int = 11,
    sip_degree: int = 3,
) -> tuple[WCS, np.ndarray]:
    """Fit an `astropy.wcs.WCS` to an `lsst.afw.geom.SkyWcs`.

    Parameters
    ----------
    wcs : `lsst.afw.geom.SkyWcs`
        The LSST WCS to fit.
    bbox : `lsst.geom.Box2I`
        The bounding box in pixel coordinates over which to fit the WCS.
    num_points : `int`, optional
        The number of points to sample along each axis for the fit.
    sip_degree : `int`, optional
        The degree of the SIP distortion to fit.

    Returns
    -------
    astropy_wcs : `astropy.wcs.WCS`
        The fitted Astropy WCS.
    residuals_arcsec : `np.ndarray`
        The residuals of the fit in arcseconds.
    """
    xs = np.linspace(bbox.getMinX(), bbox.getMaxX(), num_points)
    ys = np.linspace(bbox.getMinY(), bbox.getMaxY(), num_points)
    pixel_x, pixel_y = np.meshgrid(xs, ys)
    pixel_x = pixel_x.ravel()
    pixel_y = pixel_y.ravel()
    sky_ra, sky_dec = wcs.pixelToSkyArray(pixel_x, pixel_y, degrees=True)

    sky = SkyCoord(sky_ra * u.deg, sky_dec * u.deg, frame="icrs")
    astropy_pixel_x = pixel_x + 1
    astropy_pixel_y = pixel_y + 1
    astropy_pixel_xy = (astropy_pixel_x, astropy_pixel_y)
    astropy_wcs = fit_wcs_from_points(astropy_pixel_xy, sky, projection="TAN", sip_degree=sip_degree)

    astropy_sky = pixel_to_skycoord(astropy_pixel_x, astropy_pixel_y, astropy_wcs)
    residuals_arcsec = astropy_sky.separation(sky).arcsec

    return (astropy_wcs, residuals_arcsec)


def stitch_exposures(exposures: list[ExposureF], overlap: str = "overwrite", ramp_width: int = 32):
    """Stitch aligned ``ExposureF`` objects into a single larger exposure.

    Parameters
    ----------
    exposures : `list` [`lsst.afw.image.ExposureF`]
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
    exposures = list(exposures)
    if not exposures:
        raise ValueError("exposures must contain at least one ExposureF")

    valid_modes = {"overwrite", "average", "ramp"}
    if overlap not in valid_modes:
        raise ValueError(f"overlap must be one of {sorted(valid_modes)}, got {overlap!r}")
    if ramp_width <= 0:
        raise ValueError("ramp_width must be positive")

    min_x = min(exp.getBBox().getMinX() for exp in exposures)
    min_y = min(exp.getBBox().getMinY() for exp in exposures)
    max_x = max(exp.getBBox().getMaxX() for exp in exposures)
    max_y = max(exp.getBBox().getMaxY() for exp in exposures)
    out_bbox = Box2I(Point2I(min_x, min_y), Point2I(max_x, max_y))

    first = exposures[0]
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
        for exp in exposures:
            stitched.maskedImage.assign(exp.maskedImage, exp.getBBox())
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

    for exp in exposures:
        bbox = exp.getBBox()
        ys, xs = _bbox_slices(bbox)
        image = exp.image.array.astype(np.float32, copy=False)
        mask = exp.mask.array
        variance = exp.variance.array.astype(np.float32, copy=False)

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
