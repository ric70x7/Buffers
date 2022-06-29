import pytest
import numpy as np
import geopandas as gpd
from shapely import geometry
from buffers import clouds, buffer, regular_polygons, voronoi_polygons, disjoint_polygons

# Inputs
X = np.random.normal(0, 1, 20).reshape(-1, 2)
vrad = np.linspace(.1, 1, 10)

def test_clouds():
    with pytest.raises(TypeError):
        clouds(X, radius=.1)
    with pytest.raises(ValueError):
        clouds(X, radius=[1])
    with pytest.raises(TypeError):
        clouds(X, radius=[1, 2], num_points=3.)

    X_clouds =  clouds(X, radius=[1, 2], num_points=8)
    assert isinstance(X_clouds, list)
    assert np.all([isinstance(xci, np.ndarray) for xci in X_clouds])

def test_buffer():
    b1 =  buffer(X, radius=[1, 1], num_points=8)
    b2 =  buffer(X, radius=[1, 1], num_points=8)
    assert isinstance(b1, geometry.Polygon)
    assert isinstance(b2, geometry.Polygon)

def test_regular_polygons():
    rp =  regular_polygons(X, radius=[2, 2], num_angles=8)
    assert isinstance(rp, gpd.GeoDataFrame)
    assert X.shape[0] == rp.shape[0]

def test_vornoi_polygons():
    vp1 = voronoi_polygons(X, radius=None)
    vp2 = voronoi_polygons(X, radius=[2, 3])
    assert isinstance(vp1, gpd.GeoDataFrame)
    assert isinstance(vp2, gpd.GeoDataFrame)

def test_disjoint_polygons():
    with pytest.raises(TypeError):
        disjoint_polygons(X, radius=None)
    with pytest.raises(AssertionError):
        disjoint_polygons(X, radius=[0, 1])

    dp =  disjoint_polygons(X, radius=[2, 2], num_angles=8)
    assert isinstance(dp, gpd.GeoDataFrame)
    assert X.shape[0] == dp.shape[0]
