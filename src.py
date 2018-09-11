from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
import shapely.geometry as geometry
from matplotlib.path import Path
from shapely.geometry import Point
from shapely.affinity import rotate

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def generate_one_d_image(polyg, nx=64,ny=64):
    poly_verts = list(np.array(polyg.exterior.xy).T)

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))
    return(grid.sum(axis=0).astype('uint8'))

def gen_rand_poly_images(n = 1000,img_size = 64, n_point = 60, alpha = .2):
    #randomizes the seeds in each worker process
    np.random.seed()
    
    #generate objects at the center of the image
    a = np.random.uniform(img_size // 4, (img_size * 3) // 4, size=(n_point,2))
    points = [Point(a[i]) for i in range(n_point)]
    concave_hull, edge_points = alpha_shape(points, alpha=alpha)
    
    #if the result is a multipolygon, select the first polygon
    t_poly = concave_hull.buffer(1)
    if(type(t_poly) == geometry.multipolygon.MultiPolygon):
        t_poly = t_poly[0]
    
    rotation_angles = np.hstack((np.array([0.],dtype='float16'), np.random.uniform(0,360, size = n-1).astype('float16')))
    rotated_polygons = [rotate(t_poly,k) for k in rotation_angles]
    one_d_images = np.array([generate_one_d_image(rotated_polygons[k]) for k in range(n)])
    
    return(rotation_angles, one_d_images)