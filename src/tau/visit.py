"""
Tools which manipulate visit-level imaging.
"""

from math import ceil
from typing import Any, Iterable

import numpy as np
from lsst.afw.cameraGeom import FOCAL_PLANE, Camera
from lsst.afw.cameraGeom.utils import getCcdInCamBBoxList
from lsst.afw.geom import makeSkyWcs
from lsst.afw.image import LOCAL, ExposureF, MaskedImageF, makeExposure
from lsst.afw.math import binImage, rotateImageBy90
from lsst.daf.butler import Butler, DataId, DatasetRef, DatasetType
from lsst.geom import Box2D, Box2I, Point2I

__all__ = ["Visit", "make_visit_mosaic", "bin_exposure"]


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

    # Correct WCS for binning.
    wcs = exposure.getWcs()
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

    binned_exposure = makeExposure(exposureMI_binned)
    binned_exposure.setInfo(exposure.getInfo())
    binned_exposure.setWcs(wcs_binned)

    return binned_exposure


def make_visit_mosaic(
    detectors: dict[int, ExposureF],
    camera: Camera,
    binsize: int = 8,
    buffer: int = 10,
    background_value: float = np.nan,
    background_masks: list[str] = ["NO_DATA"],
    crop: bool = False,
) -> MaskedImageF:
    """Make a camera image from a set of detectors.

    Parameters
    ----------
    detectors : dict[int, `~lsst.afw.image.ExposureF`]
        A dictionary of detector IDs and binned exposures.
    camera : `~lsst.afw.cameraGeom.Camera`
        The camera to use.
    binsize : int, optional
        The bin size to use (should match the binsize used for the detectors).
    buffer : int, optional
        The buffer size in pixels around the camera image.
    background_value : float, optional
        The background value to use.
    background_masks : list[str], optional
        The background masks to use.
    crop : bool, optional
        Crop the camera image to the bounding box of the detectors?

    Returns
    -------
    cameraMI : `~lsst.afw.image.MaskedImageF`
        The camera image with the detectors placed.
    """

    # Get camera detectors and the camera bounding box
    if crop:
        camera_bbox_fp = Box2D()
        camera_detectors = []
        for detector_id in detectors.keys():
            camera_detectors.append(camera[detector_id])
            for corner in camera[detector_id].getCorners(FOCAL_PLANE):
                camera_bbox_fp.include(corner)
    else:
        camera_detectors = [camera[detector_id] for detector_id in detectors.keys()]
        camera_bbox_fp = camera.getFpBBox()

    # Generate a camera BBox in pixels, with a buffer
    pixel_size = camera_detectors[0].getPixelSize()
    pixel_min = Point2I(
        int(camera_bbox_fp.getMinX() // pixel_size.getX()), int(camera_bbox_fp.getMinY() // pixel_size.getY())
    )
    pixel_max = Point2I(
        int(camera_bbox_fp.getMaxX() // pixel_size.getX()), int(camera_bbox_fp.getMaxY() // pixel_size.getY())
    )
    camera_bbox = Box2I(pixel_min, pixel_max)
    camera_bbox.grow(buffer * binsize)
    camera_bbox_origin = camera_bbox.getMin()

    # Create the image, set the background
    cameraMI = MaskedImageF(
        int(ceil(camera_bbox.getDimensions().getX() / binsize)),
        int(ceil(camera_bbox.getDimensions().getY() / binsize)),
    )
    cameraMI.image[:] = background_value
    cameraMI.variance[:] = background_value

    # Loop over binned detectors, rotating and placing them in the camera image
    camera_detectors_bbox = getCcdInCamBBoxList(camera_detectors, binsize, pixel_size, camera_bbox_origin)
    for (detector_id, detector_exposure), camera_detector, camera_detector_bbox in zip(
        detectors.items(), camera_detectors, camera_detectors_bbox
    ):
        detectorMI = detector_exposure.getMaskedImage().clone()
        detectorMI = rotateImageBy90(detectorMI, camera_detector.getOrientation().getNQuarter())
        detectorMI_image = detectorMI.getImage()
        detectorMI_mask = detectorMI.getMask()
        detectorMI_variance = detectorMI.getVariance()
        # Set masked planes to background value
        bad = detectorMI_mask.getArray() & detectorMI_mask.getPlaneBitMask(background_masks) > 0
        detectorMI_image.getArray()[bad] = background_value
        # Map detector image to camera image
        camera_detectorMI = cameraMI.Factory(cameraMI, camera_detector_bbox, LOCAL)
        camera_detectorMI.setImage(detectorMI_image)
        camera_detectorMI.setMask(detectorMI_mask)
        camera_detectorMI.setVariance(detectorMI_variance)

    return cameraMI


class Visit:
    """A class to build visit mosaics."""

    def __init__(
        self,
        butler: Butler,
        dataset_type: str | DatasetType = "calexp",
        collections: str | Iterable[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Visit class.

        Parameters
        ----------
        butler : Butler
            The butler to use.
        dataset_type : str | DatasetType, optional
            The dataset type to query.
        collections : str | iterable[str], optional
            The collections to query. If None, all collections are queried.
        kwargs : Any
            Additional keyword arguments to pass to the butler.
        """
        self.butler = butler
        self.dataset_type = dataset_type
        self.collections = collections
        self.kwargs = kwargs
        self._dataset_refs = self._query_datasets(**kwargs)
        self._data_ids = [dataset_ref.dataId for dataset_ref in self._dataset_refs]
        self.detectors = [data_id["detector"] for data_id in self._data_ids]
        self.dataset_refs = {
            detector: dataset_ref for detector, dataset_ref in zip(self.detectors, self._dataset_refs)
        }
        self.data_id = self._get_visit_data_id(self._data_ids)
        self.data_ids = {detector: data_id for detector, data_id in zip(self.detectors, self._data_ids)}
        self.run = self._dataset_refs[0].run
        self.info = butler.get(self._dataset_refs[0].makeComponentRef("visitInfo"))
        self.ra = float(self.info.boresightRaDec.getRa().asDegrees())
        self.dec = float(self.info.boresightRaDec.getDec().asDegrees())
        self.summary = self.get_summary(
            self.data_id, self.ra, self.dec, self.dataset_type, self.info.exposureTime, self.run
        )
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
        ra: float | None = None,
        dec: float | None = None,
        dataset_type: str | DatasetType | None = None,
        exposure_time: float | None = None,
        run: str | None = None,
    ) -> str:
        """Generate a visit title string.

        Parameters are optional, defaulting to class attributes if not given.

        Parameters
        ----------
        data_id : dict | `~lsst.daf.butler.DataId`, optional
            The data ID of the visit.
        ra : float, optional
            The boresight right ascension of the visit, in degrees.
        dec : float, optional
            The boresight declination of the visit, in degrees.
        dataset_type : str | `~lsst.daf.butler.DatasetType`, optional
            The dataset type to query.
        dataset_count : int, optional
            The number of datasets in the visit.
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
        if not ra:
            ra = self.ra
        if not dec:
            dec = self.dec
        if not dataset_type:
            dataset_type = self.dataset_type
        if not exposure_time:
            exposure_time = self.info.exposureTime
        if not run:
            run = self.run

        # Convert to required formats.
        if not isinstance(data_id, dict):
            data_id = data_id.to_simple().dataId
        if not isinstance(dataset_type, str):
            dataset_type = dataset_type.name

        visit_id = f"Visit {data_id['visit']}" if "visit" in data_id else f"Exposure {data_id['exposure']}"
        visit_boresight = f"{ra:.2f}".replace(".", "°") + f", {dec:+.2f}".replace(".", "°")
        visit_info = f"{data_id['instrument']} {visit_id} ({visit_boresight})"
        data_info = f"data: {dataset_type}, {data_id['band']}-band, {exposure_time:.1f}s"
        run_info = f"run: {run}"

        return f"{visit_info}\n{data_info}\n{run_info}"

    def get_detector(self, detector: int, binsize: int = 1) -> ExposureF:
        """Get the dataset for a given detector number, optionally binned.

        Parameters
        ----------
        detector : int
            The detector number.
        binsize : int, optional
            Factor by which the exposure will be binned (default, no binning).

        Returns
        -------
        exposure : `~lsst.afw.image.ExposureF`
            The dataset for a given detector.
        """
        if detector not in self.detectors:
            raise ValueError(f"Detector {detector} not found in visit. Available detectors: {self.detectors}")
        exposure = self.butler.get(self.dataset_refs[detector])
        if binsize != 1:
            return bin_exposure(exposure, binsize)
        else:
            return exposure

    def make_mosaic(
        self,
        binsize: int = 8,
        camera: Camera = None,
        detectors: Iterable[int] | None = None,
        buffer: int = 10,
        background_value: float = np.nan,
        background_masks: list[str] = ["NO_DATA"],
        crop: bool = False,
    ) -> MaskedImageF:
        """Generate a mosaic of all detectors in the visit, optionally binned.

        Parameters
        ----------
        binsize : int, optional
            Factor by which the exposure will be binned.
        camera : `~lsst.afw.cameraGeom.Camera`, optional
            The camera to use. If None, an attempt is made to get the camera
            from the butler using the visit data ID and collection.
        detectors : iterable[int] | None, optional
            The detectors to include in the mosaic.
            If None, all detectors in the visit are included.
        buffer : int, optional
            The buffer size in pixels around the camera image.
        background_value : float, optional
            The background value to use.
        background_masks : list[str], optional
            The background masks to use.
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
        if not detectors:
            detectors = self.detectors
        dataset_data = {detector: self.get_detector(detector, binsize) for detector in detectors}
        return make_visit_mosaic(
            detectors=dataset_data,
            camera=camera,
            binsize=binsize,
            buffer=buffer,
            background_value=background_value,
            background_masks=background_masks,
            crop=crop,
        )
