"""
Tools used to query objects on the sky.
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad


def query_simbad(object_name, fields=["ra", "dec"]):
    """Query Simbad for object properties.

    List all available fields in the Simbad database using:

        >>> from astroquery.simbad import Simbad
        >>> Simbad.list_votable_fields()

    Parameters
    ----------
    object_name : `str`
        The name of the object to query.

    Returns
    -------
    coord : `~astropy.coordinates.SkyCoord`
        The coordinates of the object.
    """
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields(fields)
    result = custom_simbad.query_object(object_name)
    result = result[fields]
    return result


result = Simbad.query_object("Trifid Nebula")
ra = result["ra"][0]
dec = result["dec"][0]

coord = SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))
print(coord.ra.deg, coord.dec.deg)
