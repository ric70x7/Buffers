import numpy as np
import geopandas as gpd
from shapely import geometry
from shapely.ops import polygonize
from scipy.spatial import Voronoi, ConvexHull


def clouds(X: np.ndarray, radius: list, num_points: float=8):
    '''
    Create a set of points around each point in a 2D array X.

    Args:
        param X: Array of points with shape (n, 2).
        param radius: Distance from each point in the cloud to the point Xi.
        param num_points: Number of points to return for each point in X.

    Returns:
        List of 2D arrays
    '''
    vertex = np.pi * np.linspace(0, 2, num_points + 1)

    if len(radius) != 2:
        raise ValueError('radius length must be 2.')

    X_clouds = [
        np.vstack(
            [
                xi + np.array(
                    [
                        radius[0] * np.cos(t), radius[1] * np.sin(t)
                    ]
                ) for t in vertex
            ]
        ) for xi in X
    ]
    return X_clouds


def buffer(X: np.ndarray, radius: list=None, num_points: int=12):
    '''
    Define a buffer zone around a set of points X.

    Args:
        param X: Array of points with shape (n, 2).
        param radius: Minumum distance from the points to the buffer boundary.
        If radius is None the distance is zero.
        param num_points: Number of points to return for each point in X.
        Default value is 12.

    Returns:
        A shapely polygon
    '''

    if radius is not None:
        X_cloud = np.vstack(clouds(X, radius=radius, num_points=num_points))
        X = np.vstack([X, X_cloud])
    convex_hull = ConvexHull(X)
    points = convex_hull.points[convex_hull.vertices]
    polygon = geometry.Polygon(points)
    return polygon


def regular_polygons(X: np.ndarray, radius: list, num_angles: int=8):
    '''
    Define a set of regular polygons around the points X.

    Args:
        param X: Array of points with shape (n, 2).
        param radius: Circumradius of the polygon.
        param num_angles: Number of angles of each polygon. Must be >= 3.

    Returns:
        Geopandas data frame.
    '''
    polygons = clouds(X, radius=radius, num_points=num_angles)
    return gpd.GeoDataFrame(
        {
            'geometry':
            geometry.MultiPolygon([geometry.Polygon(pi) for pi in polygons])
        }
    )


def voronoi_polygons(X: np.ndarray, radius: list = None):
    '''
    Define a set of Voronoi polygons based on a set of points X.

    Args:
        param X: Array of points with shape (n, 2).
        param radius: Minumum distance from the points to the polygons.

    Returns:
        Geopandas data frame.
    '''
    if X.shape[1] != 2:
        raise ValueError('Function only implemented for 2D spaces.')

    if radius is None:
        margin = [0, 0]
    elif len(radius) != 2:
        raise ValueError('radius length must be 2.')
    else:
        margin = radius

    num_points = X.shape[0]

    c1, c2 = np.sort(X[:, 0]), np.sort(X[:, 1])
    _diffs = np.array(
        [
            max(margin[0], np.diff(c1).mean()),
            max(margin[1], np.diff(c2).mean())
        ]
    )

    min_c1, min_c2 = X.min(0) - _diffs
    max_c1, max_c2 = X.max(0) + _diffs

    box = np.vstack([
        np.vstack([np.repeat(min_c1, num_points), c2]).T,
        np.vstack([np.repeat(max_c1, num_points), c2]).T,
        np.vstack([c1, np.repeat(min_c2, num_points)]).T,
        np.vstack([c1, np.repeat(max_c2, num_points)]).T
    ])

    _X = np.vstack([X, box])

    # Define polygons geometry based on tessellation
    vor = Voronoi(_X)
    lines = [geometry.LineString(vor.vertices[li]) for li in vor.ridge_vertices if -1 not in li]
    disord = geometry.MultiPolygon(list(polygonize(lines)))
    ix_order = np.array([[i for i, di in enumerate(disord.geoms) if di.contains(geometry.Point(pi))]
                         for pi in X]).ravel()

    polygons = gpd.GeoDataFrame({'geometry': geometry.MultiPolygon([disord.geoms[i] for i in ix_order])})

    mask = buffer(X, radius=radius)
    polygons = polygons.clip(mask)

    return polygons.sort_index()


def disjoint_polygons(X: np.ndarray, radius: list, num_angles: int=8):
    '''
    Define a set of non overlapping polygons around the points X.

    Args:
        param X: Array of points with shape (n, 2).
        param radius: Circumradius of the polygon.
        param num_angles: Number of angles of each polygon. Must be >= 3.

    Returns:
        Geopandas data frame.

    :return: Geopandas data frame.
    '''
    assert np.all([ri > 0 for ri in radius]), 'radius values must be positive'
    vorpol = voronoi_polygons(X, radius=[2*ri for ri in radius])
    regpol = regular_polygons(X, radius=radius, num_angles=num_angles)
    dispol = [
        vi.intersection(pi) for vi,pi in zip(vorpol.geometry, regpol.geometry)
    ]
    return gpd.GeoDataFrame({'geometry': geometry.MultiPolygon(dispol)})
