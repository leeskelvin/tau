"""
Tools which manipulate visit-level imaging.
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import ceil
from typing import Any

import numpy as np
from lsst.afw.cameraGeom import FOCAL_PLANE, Camera, Detector
from lsst.afw.cameraGeom.utils import getCcdInCamBBoxList
from lsst.afw.geom import makeSkyWcs
from lsst.afw.image import LOCAL, ExposureF, MaskedImageF, makeExposure
from lsst.afw.math import binImage, rotateImageBy90
from lsst.daf.butler import Butler, DataId, DatasetRef, DatasetType
from lsst.geom import Box2D, Box2I, Point2I
from lsst.pex.exceptions import LengthError

from .utils import ref_to_title

__all__ = [
    "make_exposure",
    "bin_exposure",
    "get_exposure_from_ref",
    "get_exposures_from_refs",
    "make_visit_mosaic",
    "Visit",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
logger.addHandler(handler)


def make_exposure(data: Any, detector: Detector | None = None) -> ExposureF:
    """Coerce input data into an ExposureF, where possible.

    Parameters
    ----------
    data : Any
        The input data from which to make the exposure.
    detector : `~lsst.afw.cameraGeom.Detector`, optional
        The detector object to associate with the data, if any.

    Returns
    -------
    exposure : `~lsst.afw.image.ExposureF`
        The extracted exposure.
    """
    if not isinstance(data, ExposureF):
        if hasattr(data, "getImage"):
            exposure = ExposureF(MaskedImageF(data.getImage()))
        else:
            raise ValueError("Dataset is not an ExposureF, and cannot be coerced into one.")
    else:
        exposure = data
    if detector is not None:
        exposure.setDetector(detector)
    return exposure


def bin_exposure(exposure: ExposureF, binsize: int) -> ExposureF:
    """Bin an exposure into a given bin size.

    Parameters
    ----------
    exposure : ExposureF
        The exposure to bin.
    binsize : int
        The bin size.

    Returns
    -------
    binned_exposure : ExposureF
        The binned exposure.
    """
    # Raise if binsize is not a positive integer.
    if not isinstance(binsize, int) or binsize < 1:
        raise ValueError("binsize must be a positive integer.")

    exposureMI = exposure.getMaskedImage()
    exposureMI_binned = binImage(exposureMI, binsize)

    binned_exposure = makeExposure(exposureMI_binned)
    binned_exposure.setInfo(exposure.getInfo())

    # Correct WCS for binning.
    wcs = exposure.getWcs()
    if wcs is not None:
        xy = wcs.getPixelOrigin().clone()
        xy.setX(xy[0] / binsize)
        xy.setY(xy[1] / binsize)
        ad = wcs.getSkyOrigin()
        cd = wcs.getCdMatrix().copy() * binsize
        wcs_binned = makeSkyWcs(
            crpix=xy,
            crval=ad,
            cdMatrix=cd,
        )
        binned_exposure.setWcs(wcs_binned)

    return binned_exposure


def get_exposure_from_ref(
    dataset_ref: DatasetRef, butler: Butler, binsize: int = 1, camera: Camera | None = None
) -> ExposureF:
    """Get the exposure for a given dataset reference, optionally binned.

    Parameters
    ----------
    dataset_ref : `~lsst.daf.butler.DatasetRef`
        Dataset reference for the exposure to retrieve.
    butler : `~lsst.daf.butler.Butler`
        Butler to use to retrieve the exposure.
    binsize : int, optional
        Factor by which the exposure will be binned (default, no binning).
    camera : `~lsst.afw.cameraGeom.Camera`, optional
        Camera to use. Only used to supply detector information when
        the dataset is not an ExposureF or when confirming existing binning.

    Returns
    -------
    exposure : `~lsst.afw.image.ExposureF`
        Exposure for the requested dataset ref, optionally binned.
    """
    try:
        exposure = butler.get(dataset_ref)
    except FileNotFoundError:
        logger.error(f"Dataset not found for {dataset_ref}.")
        return None
    detector_id = dict(dataset_ref.dataId.mapping).get("detector", None)
    if not isinstance(exposure, ExposureF):
        if detector_id is not None:
            assert camera is not None and isinstance(camera, Camera)
            exposure = make_exposure(exposure, camera[detector_id])
        else:
            exposure = make_exposure(exposure)
    if not exposure.getDetector() and detector_id is not None:
        assert camera is not None and isinstance(camera, Camera)
        exposure.setDetector(camera[detector_id])
    if binsize != 1:
        if detector_id is not None:
            assert camera is not None and isinstance(camera, Camera)
            camera_y, camera_x = camera[detector_id].getBBox().getDimensions()
            detector_y, detector_x = exposure.getBBox().getDimensions()
            native_binsize = int(np.round(np.mean([camera_x / detector_x, camera_y / detector_y])))
            if native_binsize != 1:
                logger.info(
                    f"Using already binned image binsize of {native_binsize} for detector {detector_id}."
                )
                return exposure
        return bin_exposure(exposure, binsize)
    else:
        return exposure


def get_exposures_from_refs(
    dataset_refs: Sequence[DatasetRef],
    butler: Butler,
    binsize: int = 1,
    camera: Camera | None = None,
    max_workers: int = 8,
) -> list[ExposureF]:
    """Get exposures for a list of dataset references, optionally binned.

    Parameters
    ----------
    butler : `~lsst.daf.butler.Butler`
        Butler to use to retrieve the exposures.
    dataset_refs : Sequence[`~lsst.daf.butler.DatasetRef`]
        Dataset references for the exposures to retrieve.
    binsize : int, optional
        Factor by which the exposures will be binned (default, no binning).
    camera : `~lsst.afw.cameraGeom.Camera`, optional
        Camera to use. Only used to supply detector information when
        the dataset is not an ExposureF or when confirming existing binning.

    Returns
    -------
    exposures : list[`~lsst.afw.image.ExposureF`]
        List of exposures for the requested dataset refs, optionally binned.
    """
    if max_workers <= 1:
        return [get_exposure_from_ref(dataset_ref, butler, binsize, camera) for dataset_ref in dataset_refs]
    else:
        get_exposure_from_ref_partial = partial(
            get_exposure_from_ref, butler=butler, binsize=binsize, camera=camera
        )
        with ProcessPoolExecutor(max_workers=max(len(dataset_refs), max_workers)) as executor:
            exposures = list(executor.map(get_exposure_from_ref_partial, dataset_refs))
        return exposures


def make_visit_mosaic(
    exposures: list[ExposureF],
    camera: Camera,
    buffer: int = 10,
    background_value: float = np.nan,
    background_masks: list[str] = ["NO_DATA"],
    crop: bool = False,
) -> MaskedImageF:
    """Make a camera image from a set of detector exposures.

    Parameters
    ----------
    exposures : list[~lsst.afw.image.ExposureF]
        Exposures to use, already binned if desired.
    camera : `~lsst.afw.cameraGeom.Camera`
        Camera to use.
    buffer : int, optional
        Buffer size in pixels around the camera image.
    background_value : float, optional
        Background value to use.
    background_masks : list[str], optional
        Background masks to use.
    crop : bool, optional
        Crop the camera image to the bounding box of the detectors?

    Returns
    -------
    cameraMI : `~lsst.afw.image.MaskedImageF`
        The camera image with the detectors placed.
    """
    # Get camera detectors and the camera bounding box
    detector_ids = [exposure.getDetector().getId() for exposure in exposures]
    if crop:
        camera_detectors = []
        camera_bbox_fp = Box2D()
        for detector_id in detector_ids:
            camera_detectors.append(camera[detector_id])
            for corner in camera[detector_id].getCorners(FOCAL_PLANE):
                camera_bbox_fp.include(corner)
    else:
        camera_detectors = [camera[detector_id] for detector_id in detector_ids]
        camera_bbox_fp = camera.getFpBBox()

    # Get binsize
    camera_y, camera_x = camera_detectors[0].getBBox().getDimensions()
    detector_y, detector_x = exposures[0].getBBox().getDimensions()
    binsize = int(np.round(np.mean([camera_x / detector_x, camera_y / detector_y])))

    # Generate a camera BBox in pixels, with a buffer
    pixel_size = camera_detectors[0].getPixelSize()
    pixel_min_x = int(camera_bbox_fp.getMinX() // pixel_size.getX())
    pixel_min_y = int(camera_bbox_fp.getMinY() // pixel_size.getY())
    pixel_min = Point2I(pixel_min_x, pixel_min_y)
    pixel_max_x = int(camera_bbox_fp.getMaxX() // pixel_size.getX())
    pixel_max_y = int(camera_bbox_fp.getMaxY() // pixel_size.getY())
    pixel_max = Point2I(pixel_max_x, pixel_max_y)
    camera_bbox = Box2I(pixel_min, pixel_max)
    camera_bbox.grow(buffer * binsize)
    camera_bbox_origin = camera_bbox.getMin()

    # Create the image, set the background
    cameraMI_width = int(ceil(camera_bbox.getDimensions().getX() / binsize))
    cameraMI_height = int(ceil(camera_bbox.getDimensions().getY() / binsize))
    cameraMI = MaskedImageF(cameraMI_width, cameraMI_height)
    cameraMI.image[:] = background_value
    cameraMI.variance[:] = background_value

    # Loop over binned detectors, rotating and placing them in the camera image
    camera_detectors_bbox = getCcdInCamBBoxList(camera_detectors, binsize, pixel_size, camera_bbox_origin)
    for detector_id, exposure, camera_detector, camera_detector_bbox in zip(
        detector_ids, exposures, camera_detectors, camera_detectors_bbox
    ):
        detectorMI = exposure.getMaskedImage().clone()
        detectorMI = rotateImageBy90(detectorMI, camera_detector.getOrientation().getNQuarter())
        detectorMI_image = detectorMI.getImage()
        detectorMI_mask = detectorMI.getMask()
        detectorMI_variance = detectorMI.getVariance()
        # Set masked planes to background value
        bad = detectorMI_mask.getArray() & detectorMI_mask.getPlaneBitMask(background_masks) > 0
        detectorMI_image.getArray()[bad] = background_value
        # Map detector image to camera image
        camera_detectorMI = cameraMI.Factory(cameraMI, camera_detector_bbox, LOCAL)
        try:
            camera_detectorMI.setImage(detectorMI_image)
            camera_detectorMI.setMask(detectorMI_mask)
            camera_detectorMI.setVariance(detectorMI_variance)
        except LengthError:
            logger.warning(
                f"Data for detector {detector_id} does not fit in the detector-in-camera BBox dimensions "
                f"(data: {detectorMI_image.getBBox().getDimensions()}, "
                f"camera: {camera_detector_bbox.getDimensions()}), skipping."
            )
            continue

    return cameraMI


class Visit:
    """A class to build visit mosaics."""

    def __init__(
        self,
        butler: Butler,
        dataset_type: str | DatasetType = "calexp",
        collections: str | Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Visit class.

        Parameters
        ----------
        butler : Butler
            The butler to use.
        dataset_type : str | DatasetType, optional
            The dataset type to query.
        collections : str | Sequence[str], optional
            The collections to query. If None, all collections are queried.
        kwargs : Any
            Additional keyword arguments to pass to the butler.
        """
        self.butler = butler
        self.dataset_type = dataset_type
        self.collections = collections
        self.kwargs = kwargs
        self._dataset_refs = self._query_datasets(**kwargs)
        data_ids = [dataset_ref.dataId for dataset_ref in self._dataset_refs]
        if "visit" in data_ids[0]:
            unique_visits = list({data_id["visit"] for data_id in data_ids})
            if len(unique_visits) > 1:
                examples = str(unique_visits[:5])[1:-1] + ("..." if len(unique_visits) > 5 else "")
                raise ValueError(f"Multiple visits found in query: {examples}")
        elif "exposure" in data_ids[0]:
            unique_exposures = list({data_id["exposure"] for data_id in data_ids})
            if len(unique_exposures) > 1:
                examples = str(unique_exposures[:5])[1:-1] + ("..." if len(unique_exposures) > 5 else "")
                raise ValueError(f"Multiple exposures found in query: {examples}")
        self._dataset_refs.sort(key=lambda dataset_ref: dataset_ref.dataId["detector"])
        self._data_ids = [dataset_ref.dataId for dataset_ref in self._dataset_refs]
        self.detector_ids = [data_id["detector"] for data_id in self._data_ids]
        self.dataset_refs = dict(zip(self.detector_ids, self._dataset_refs))
        self.data_id = self._get_visit_data_id(self._data_ids)
        self.data_ids = dict(zip(self.detector_ids, self._data_ids))
        self.run = self._dataset_refs[0].run
        try:
            self.info = butler.get(self._dataset_refs[0].makeComponentRef("visitInfo"))
            self.ra = float(self.info.boresightRaDec.getRa().asDegrees())
            self.dec = float(self.info.boresightRaDec.getDec().asDegrees())
        except KeyError:
            self.info = None
            self.ra = None
            self.dec = None
        self.summary = self.get_summary(self.data_id, self.dataset_type, self.run)
        self.title = ref_to_title(self._dataset_refs[0], exclude=["detector"])
        self.camera = self._get_camera()

    def __str__(self) -> str:
        """Provide a string representation of the Visit instance."""
        return self.summary

    def __getitem__(self, index: int) -> DatasetRef:
        """Retrieve the dataset reference for a given detector ID.

        Parameters
        ----------
        index : int
            The index of the dataset reference to retrieve.

        Returns
        -------
        dataset_ref : `~lsst.daf.butler.DatasetRef`
            The dataset reference.
        """
        return self._dataset_refs[index]

    def __len__(self) -> int:
        """Return the number of detectors in the visit.

        Returns
        -------
        detector_count : int
            The number of detectors in the visit.
        """
        return len(self.dataset_refs)

    def _query_datasets(self, **kwargs: Any) -> list:
        """Query the butler for datasets.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments to pass to butler.query_datasets.

        Returns
        -------
        dataset_refs : list [`~lsst.daf.butler.DatasetRef`]
            The query results.
        """
        return self.butler.query_datasets(
            dataset_type=self.dataset_type,
            collections=self.collections,
            **kwargs,
        )

    def _get_visit_data_id(self, data_ids: list[DataId]) -> DataId:
        """Get the data ID common to all datasets in the visit.

        Parameters
        ----------
        data_ids : list[`~lsst.daf.butler.DataId`]
            A list of data IDs to combine.

        Returns
        -------
        visit_data_id : `~lsst.daf.butler.DataId`
            The combined data ID.
        """
        visit_data_id = data_ids[0]
        dimensions = visit_data_id.dimensions.to_simple()
        dimensions.remove("detector")
        return visit_data_id.subset(dimensions)

    def _get_camera(self) -> Camera | None:
        """Attempt to get the camera for the visit using the instantiation
        collections and instrument from the visit data ID.

        None is returned if the camera cannot be found.

        Returns
        -------
        camera : `~lsst.afw.cameraGeom.Camera` | None
            The camera for the visit, if found.
        """
        try:
            camera_refs = list(
                set(
                    self.butler.query_datasets(
                        "camera",
                        instrument=self.data_id["instrument"],
                        collections=self.collections,
                        find_first=False,
                    )
                )
            )
            return self.butler.get(camera_refs[0])
        except Exception:
            return None

    def get_summary(
        self,
        data_id: dict | DataId | None = None,
        dataset_type: str | DatasetType | None = None,
        run: str | None = None,
    ) -> str:
        """Generate a visit title string.

        Parameters are optional, defaulting to class attributes if not given.

        Parameters
        ----------
        data_id : dict | `~lsst.daf.butler.DataId`, optional
            The data ID of the visit.
        dataset_type : str | `~lsst.daf.butler.DatasetType`, optional
            The dataset type to query.
        run : str, optional
            The run to query.

        Returns
        -------
        summary : str
            Information string summarizing this visit.
        """
        # Set parameters to class attributes if not given.
        if not data_id:
            data_id = self.data_id
        if not dataset_type:
            dataset_type = self.dataset_type
        if not run:
            run = self.run

        # Convert to required formats.
        if not isinstance(data_id, dict):
            data_id = data_id.to_simple().dataId
        if not isinstance(dataset_type, str):
            dataset_type = dataset_type.name

        match data_id:
            case {"band": value}:
                band_id = f"({value}-band)"
            case _:
                band_id = ""
        match data_id:
            case {"visit": value}:
                visit_id = f", visit {value}"
            case {"exposure": value}:
                visit_id = f", exposure {value}"
            case _:
                visit_id = ""
        data_info = f"{data_id['instrument']} {dataset_type} {band_id}{visit_id}"
        run_info = f"run: {run}"

        return f"{data_info}\n{run_info}"

    def get_exposure(self, detector_id: int, binsize: int = 1, camera: Camera | None = None) -> ExposureF:
        """Get the exposure for a given detector number, optionally binned.

        Parameters
        ----------
        detector_id : int
            Detector number ID.
        binsize : int, optional
            Factor by which the exposure will be binned (default, no binning).
        camera : `~lsst.afw.cameraGeom.Camera`, optional
            Camera to use. Only used to supply detector information when
            the dataset is not an ExposureF. If None, the camera attribute
            of the visit is used.

        Returns
        -------
        exposure : `~lsst.afw.image.ExposureF`
            The exposure for a given detector.
        """
        if camera is None:
            camera = self.camera
        if detector_id not in self.detector_ids:
            logger.error(
                f"Detector {detector_id} not found in visit. Available detectors: {self.detector_ids}"
            )
            return None
        exposure = get_exposure_from_ref(self.dataset_refs[detector_id], self.butler, binsize, camera)
        return exposure

    def get_mosaic(
        self,
        binsize: int = 8,
        camera: Camera = None,
        detector_ids: Sequence[int] | None = None,
        buffer: int = 10,
        background_value: float = np.nan,
        background_masks: list[str] = ["NO_DATA"],
        crop: bool = False,
    ) -> MaskedImageF:
        """Generate a mosaic of detectors in the visit, optionally binned.

        Parameters
        ----------
        binsize : int, optional
            Factor by which the exposure will be binned.
        camera : `~lsst.afw.cameraGeom.Camera`, optional
            Camera to use. If None, an attempt is made to get the camera
            from the butler using the visit data ID and collection.
        detector_ids : Sequence[int] | None, optional
            Detectors to include in the mosaic.
            If None, all detectors in the visit are included.
        buffer : int, optional
            Buffer size in pixels around the camera image.
        background_value : float, optional
            Background value to use.
        background_masks : list[str], optional
            Background masks to use.
        crop : bool, optional
            Crop the camera image to the bounding box of the detectors?

        Returns
        -------
        visit_mosaic : `~lsst.afw.image.ExposureF`
            The mosaic of all detectors in the visit.
        """
        if not camera:
            if not self.camera:
                raise ValueError("No camera found for visit.")
            else:
                camera = self.camera
        if not detector_ids:
            detector_ids = self.detector_ids

        binned_exposures = []
        for detector_id in detector_ids:
            binned_exposure = self.get_exposure(detector_id, binsize, camera)
            if binned_exposure is not None:
                binned_exposures.append(binned_exposure)

        return make_visit_mosaic(
            exposures=binned_exposures,
            camera=camera,
            buffer=buffer,
            background_value=background_value,
            background_masks=background_masks,
            crop=crop,
        )
