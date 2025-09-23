"""
Utilities for introspecting astronomy data.
"""

import logging
import re
import sys
import textwrap
import time

# from collections.abc import Mapping
import yaml
from lsst.daf.butler import DatasetRef
from lsst.pipe.base import Pipeline
from lsst.source.injection.utils._make_injection_pipeline import _parse_config_override
from lsst.utils.packages import getEnvironmentPackages

__all__ = ["print_session_info", "ref_to_title", "convert_selection_sets", "dump_config"]


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
    ref: DatasetRef, modifier: str = "", exclude: str | list[str] = "", delimiter: str = "\n", wrap=80
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
    data_id = {k: v for k, v in ref.dataId.required.items() if k not in exclude}
    data_id_str = ", ".join([f"{k}: {repr(v)}" for k, v in data_id.items()])
    parts = [ref.datasetType.name + modifier, data_id_str, ref.run]
    wrapped_parts = [
        textwrap.fill(p, width=wrap, break_long_words=False, break_on_hyphens=False) for p in parts
    ]
    return delimiter.join(wrapped_parts)


# def walk_and_report(obj, path="root"):
#     if isinstance(obj, Mapping):
#         for k, v in obj.items():
#             walk_and_report(v, f"{path}.{k}")
#     elif isinstance(obj, list):
#         for i, v in enumerate(obj):
#             walk_and_report(v, f"{path}[{i}]")
#     else:
#         if "lsst" in type(obj).__module__:
#             print(f"{path}\n{obj}\n{type(obj)}\n\n")


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
