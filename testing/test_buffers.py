import pytest
import geopandas as geop
from buffers import *
import pytest

# Inputs
X = np.random.normal(0, 1, 20).reshape(-1, 2)
vrad = np.linspace(.1, 1, 10)


def test_vornoi_polygons():
    with pytest.raises(AssertionError):
       voronoi_polygons(X=0)
       voronoi_polygons(X=X.flatten())
       voronoi_polygons(X=X.T)
    vorpol = voronoi_polygons(X=X)
    assert isinstance(vorpol, geop.GeoDataFrame)
    assert vorpol.shape[0] == X.shape[0]


def test_regular_polygons():
    with pytest.raises(AssertionError):
        regular_polygons(X.flatten(), radius=.1)
        regular_polygons(X.T, radius=.1)
        regular_polygons(X, radius=0)
        regular_polygons(X, radius=.1, n_angles=2)
        regular_polygons(X, radius=vrad[:, None], n_angles=2)
        regular_polygons(X, radius=vrad[:5], n_angles=2)

    regpol = regular_polygons(X, radius=.1, n_angles=6)
    assert isinstance(regpol, geop.GeoDataFrame)
    assert X.shape[0] == regpol.shape[0]

    regpol = regular_polygons(X, radius=vrad, n_angles=6)
    assert isinstance(regpol, geop.GeoDataFrame)
    assert X.shape[0] == regpol.shape[0]


def test_disjoint_polygons():
    dispol = disjoint_polygons(X, radius=.1, n_angles=6)
    assert isinstance(dispol, geop.GeoDataFrame)
    assert X.shape[0] == dispol.shape[0]
