from caudal.sources import nldi

# (longitude, latitude); no edge lies on a multiple of 0.125, so no lattice point sits on the outline
SQUARE = [(-123.3, 38.95), (-122.98, 38.95), (-122.98, 39.3), (-123.3, 39.3)]


def test_inside_tells_a_point_in_the_outline_from_one_beside_it():
    assert nldi.inside(SQUARE, -123.1, 39.1)
    assert not nldi.inside(SQUARE, -122.9, 39.1) and not nldi.inside(SQUARE, -123.1, 39.4)
    closed = [*SQUARE, SQUARE[0]]  # a ring that repeats its first point, as GeoJSON writes it
    assert nldi.inside(closed, -123.1, 39.1) and not nldi.inside(closed, -122.9, 39.1)


def test_lattice_points_sit_on_multiples_of_the_step():
    got = nldi.lattice(SQUARE, 0.125)
    # latitudes 39.25, 39.125 and 39.0, longitudes -123.25, -123.125 and -123.0: north to south, then west to east
    assert got == [(lat, lon) for lat in (39.25, 39.125, 39.0) for lon in (-123.25, -123.125, -123.0)]


def test_outline_takes_the_largest_part():
    small, big = [[[0, 0], [1, 0], [1, 1]]], [[[0, 0], [2, 0], [2, 2], [0, 2]]]
    geojson = {"features": [{"geometry": {"type": "MultiPolygon", "coordinates": [small, big]}}]}
    assert nldi.outline(geojson) == [(0, 0), (2, 0), (2, 2), (0, 2)]
