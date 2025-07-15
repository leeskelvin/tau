"""
Tools used to query objects on the sky.
"""

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad

__all__ = ["query_object", "query_region"]


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
    ra: float, dec: float, radius: float | str, extra_fields: list[str] = [], unit: str = "arcsec"
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
    custom_simbad.add_votable_fields(*extra_fields)
    result = custom_simbad.query_region(coordinates, radius=radius)
    return result
