"""
Tools used to query objects on the sky.
"""

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad

__all__ = ["query_object", "query_region", "query_box"]


def query_object(object_name: str, extra_fields: list[str] = [], wildcard: bool = False) -> Table:
    """Query Simbad for object properties.

    List VOTable fields in the Simbad database using:

        >>> from astroquery.simbad import Simbad
        >>> Simbad.list_votable_fields()

    Parameters
    ----------
    object_name : `str`
        The name of the object to query.
    extra_fields : `list`[`str`]
        Additional fields to include in the query. These should be valid
        VOTable field names as listed by `Simbad.list_votable_fields()`.
    wildcard : `bool`, optional
        If `True`, the object name will be treated as a wildcard search.

    Returns
    -------
    result : `~astropy.table.Table`
        A table containing the queried properties of the object.
    """
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields(*extra_fields)
    result = custom_simbad.query_object(object_name, wildcard=wildcard)
    return result


def query_region(
    ra: float,
    dec: float,
    radius: float | str,
    extra_fields: list[str] = ["g", "r", "i"],
    unit: str = "arcsec",
) -> Table:
    """Query Simbad for objects in a specified region.

    List VOTable fields in the Simbad database using:

        >>> from astroquery.simbad import Simbad
        >>> Simbad.list_votable_fields()

    Parameters
    ----------
    ra : `float`
        Right ascension of the center of the region in degrees.
    dec : `float`
        Declination of the center of the region in degrees.
    radius : `float` | `str`
        Radius of the region to query. If a `float`, units are associated with
        the `unit` parameter. If a `str`, it should be a string representation
        of a quantity (e.g., "1 arcmin").
    extra_fields : `list`[`str`]
        Additional fields to include in the query. These should be valid
        VOTable field names as listed by `Simbad.list_votable_fields()`.
    unit : `str`, optional
        The unit of the radius if it is not a string.

    Returns
    -------
    result : `~astropy.table.Table`
        A table containing the objects found in the specified region.
    """
    coordinates = SkyCoord(ra=ra, dec=dec, unit="deg")
    if not isinstance(radius, str):
        radius = Angle(float(radius), unit=unit)
    custom_simbad = Simbad()
    try:
        custom_simbad.add_votable_fields(*extra_fields)
    except ValueError as e:
        votable_fields = ", ".join(Simbad.list_votable_fields()["name"])
        raise ValueError(f"{e} Valid extra_fields are: {votable_fields}")
    result = custom_simbad.query_region(coordinates, radius=radius)
    cols_to_keep = ["main_id", "ra", "dec"] + extra_fields
    if result is not None:
        result = result[cols_to_keep]
        result.sort("ra", reverse=True)
    return result


def query_box(
    ra_lim: tuple[float, float],
    dec_lim: tuple[float, float],
    extra_fields: list[str] = [],
) -> Table:
    """Query Simbad for objects within a rectangular RA/Dec box.

    This calls `query_region` with a circle that fully encloses the box,
    then filters the results to the specified RA/Dec limits.

    Parameters
    ----------
    ra_lim : `tuple`[`float`, `float`]
        Tuple of right ascension limits of the box in degrees.
    dec_lim : `tuple`[`float`, `float`]
        Tuple of declination limits of the box in degrees.
    extra_fields : `list`[`str`]
        Additional fields to include in the query. These should be valid
        VOTable field names as listed by `Simbad.list_votable_fields()`.

    Returns
    -------
    result : `~astropy.table.Table`
        A table containing the objects found within the specified box.
        If no objects are found, an empty table is returned.
    """
    ra_center = 0.5 * (ra_lim[0] + ra_lim[1])
    dec_center = 0.5 * (dec_lim[0] + dec_lim[1])
    radius = np.sqrt((np.max(ra_lim) - ra_center) ** 2 + (np.max(dec_lim) - dec_center) ** 2)
    result = query_region(ra_center, dec_center, radius, extra_fields, unit="degree")
    if result is None:
        return Table()  # no results
    mask = (
        (result["ra"] >= np.min(ra_lim))
        & (result["ra"] <= np.max(ra_lim))
        & (result["dec"] >= np.min(dec_lim))
        & (result["dec"] <= np.max(dec_lim))
    )
    return result[mask]
