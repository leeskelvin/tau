"""
Utilities for introspecting astronomy data.
"""

import logging
import re
import sys
import textwrap
import time
from collections.abc import Sequence

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_skycoord
from lsst.afw.geom import SkyWcs
from lsst.daf.butler import DatasetRef
from lsst.geom import Box2I
from lsst.pipe.base import Pipeline
from lsst.source.injection.utils._make_injection_pipeline import _parse_config_override
from lsst.utils.packages import getEnvironmentPackages

__all__ = [
    "print_session_info",
    "ref_to_title",
    "convert_selection_sets",
    "dump_config",
    "fit_lsst_wcs",
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
    ref: DatasetRef | Sequence[DatasetRef],
    modifier: str = "",
    exclude: str | list[str] = "",
    delimiter: str = "\n",
    wrap=80,
    show_dataset_type: bool = True,
    show_data_id: bool = True,
    show_run: bool = True,
) -> str:
    """Convert one or more DatasetRefs to a title string for a plot.

    Parameters
    ----------
    ref : `DatasetRef` | `~collections.abc.Sequence` [`DatasetRef`]
        One DatasetRef, or multiple DatasetRefs to summarize.
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
    refs = [ref] if isinstance(ref, DatasetRef) else list(ref)
    if not refs:
        return ""

    if isinstance(exclude, str):
        exclude_set = {exclude} if exclude else set()
    else:
        exclude_set = set(exclude)

    def _unique(values):
        unique = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _format_values(values):
        values = _unique(values)
        if len(values) == 1:
            return repr(values[0])
        return ", ".join(repr(v) for v in values)

    parts = []
    if show_dataset_type:
        dataset_types = _unique([r.datasetType.name for r in refs])
        if len(dataset_types) == 1:
            parts.append(dataset_types[0] + modifier)
        else:
            parts.append(f"dataset_types: [{', '.join(dataset_types)}]{modifier}")

    if show_data_id:
        if len(refs) == 1:
            data_id = {k: v for k, v in refs[0].dataId.required.items() if k not in exclude_set}
            data_id_str = ", ".join([f"{k}: {repr(v)}" for k, v in data_id.items()])
        else:
            keys = []
            for r in refs:
                for key in r.dataId.required:
                    if key not in exclude_set and key not in keys:
                        keys.append(key)

            key_values = {k: [r.dataId.required.get(k) for r in refs] for k in keys}
            data_id_parts = []
            for key in keys:
                values = key_values[key]
                unique_values = _unique(values)
                formatted = repr(unique_values[0]) if len(unique_values) == 1 else _format_values(values)
                data_id_parts.append(f"{key}: {formatted}")

            data_id_str = ", ".join(data_id_parts)
        parts.append(data_id_str)

    if show_run:
        runs = _unique([r.run for r in refs])
        if len(runs) == 1:
            parts.append(runs[0])
        else:
            parts.append("runs: " + _format_values(runs))

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
