from shapely import MultiPolygon


def try_cast_multipolygon_to_polygon(geom):
    if isinstance(geom, MultiPolygon) and len(geom.geoms) == 1:
        # If the MultiPolygon contains only one Polygon, return that Polygon
        return geom.geoms[0]
    else:
        raise Exception('Cannot cast the MultiPolygon into a Polygon.')
