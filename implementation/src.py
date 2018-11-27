#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:51:16 2018

@author: serkankarakulak
"""


#from shapely.ops import cascaded_union, polygonize
#from scipy.spatial import Delaunay
import numpy as np
#import math
#import shapely.geometry as geometry
#from matplotlib.path import Path
#from shapely.geometry import Point
#from shapely.affinity import rotate
import os

#def strided_indexing_roll(a, r):
#    # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
#    # Concatenate with sliced to cover all rolls
#    a_ext = np.concatenate((a,a[:,:-1]),axis=1)
#
#    # Get sliding windows; use advanced-indexing to select appropriate ones
#    n = a.shape[1]
#    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]
#
def safe_mkdir(path):
    """ 
    Create a directory if there isn't one already. 
    """
    try:
        os.mkdir(path)
    except OSError:
        pass
#
#def alpha_shape(points, alpha):
#    """
#    Compute the alpha shape (concave hull) of a set of points.
#
#    @param points: Iterable container of points.
#    @param alpha: alpha value to influence the gooeyness of the border. Smaller
#                  numbers don't fall inward as much as larger numbers. Too large,
#                  and you lose everything.
#    """
#    if len(points) < 4:
#        # When you have a triangle, there is no sense in computing an alpha
#        # shape.
#        return geometry.MultiPoint(list(points)).convex_hull
#
#    def add_edge(edges, edge_points, coords, i, j):
#        """Add a line between the i-th and j-th points, if not in the list already"""
#        if (i, j) in edges or (j, i) in edges:
#            # already added
#            return
#        edges.add( (i, j) )
#        edge_points.append(coords[ [i, j] ])
#
#    coords = np.array([point.coords[0] for point in points])
#
#    tri = Delaunay(coords)
#    edges = set()
#    edge_points = []
#    # loop over triangles:
#    # ia, ib, ic = indices of corner points of the triangle
#    for ia, ib, ic in tri.vertices:
#        pa = coords[ia]
#        pb = coords[ib]
#        pc = coords[ic]
#
#        # Lengths of sides of triangle
#        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
#        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
#        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
#
#        # Semiperimeter of triangle
#        s = (a + b + c)/2.0
#
#        # Area of triangle by Heron's formula
#        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
#        circum_r = a*b*c/(4.0*area)
#
#        # Here's the radius filter.
#        #print circum_r
#        if circum_r < 1.0/alpha:
#            add_edge(edges, edge_points, coords, ia, ib)
#            add_edge(edges, edge_points, coords, ib, ic)
#            add_edge(edges, edge_points, coords, ic, ia)
#
#    m = geometry.MultiLineString(edge_points)
#    triangles = list(polygonize(m))
#    return cascaded_union(triangles), edge_points
#
#def generate_one_d_image(polyg, nx=64,ny=64, shift_n = 8):
#    """ 
#    Takes a two-dimentional polygon as its input,
#    and produces the polygons one-dimentional density 
#    by taking integral along the x-axis. """
#    poly_verts = list(np.array(polyg.exterior.xy).T)
#
#    # Create vertex coordinates for each grid cell...
#    # (<0,0> is at the top left of the grid in this system)
#    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
#    x, y = x.flatten(), y.flatten()
#
#    points = np.vstack((x,y)).T
#
#    path = Path(poly_verts)
#    grid = path.contains_points(points)
#    grid = grid.reshape((ny,nx))
#    
#    oneDImage = grid.sum(axis=0).astype('uint8')
#    
#    # random image translations
#    if(shift_n > 0):
#        temp_1d_np = np.zeros(nx+shift_n*2)
#        col_start = np.random.randint(-shift_n, shift_n+1) # [low,high)
#        temp_1d_np[8+col_start:8+col_start+64] = oneDImage
#        oneDImage = np.copy(temp_1d_np[8:8+64])
#    
#    return(oneDImage)
# 
#def two_d_image(polyg, nx0=64,nx1=64,o_nx0=48, o_nx1=48):
#    """ 
#    Takes a two-dimentional polygon as its input,
#    and produces 2d boolean mapping.
#    """
#    poly_verts = list(np.array(polyg.exterior.xy).T)
#
#    # Create vertex coordinates for each grid cell...
#    # (<0,0> is at the top left of the grid in this system)
#    x, y = np.meshgrid(np.arange(nx0), np.arange(nx1))
#    x, y = x.flatten(), y.flatten()
#
#    points = np.vstack((x,y)).T
#
#    path = Path(poly_verts)
#    grid = path.contains_points(points)
#    grid = grid.reshape((nx1,nx0))
#    
#    # finds the the minimum x_0 and x_1 values of the object. 
#    # lower limit exists to keep the dimension equal to o_nx
#    temp= grid.argmax(axis=0)
#    min_x0 = min(temp[temp>0].min(),nx0-o_nx0)
#    temp= grid.argmax(axis=1)
#    min_x1 = min(temp[temp>0].min(),nx1-o_nx1)
#    # returns an array sized (o_nx0, o_nx1), where we crop 
#    # out the empty rows at the top, and the empty columns
#    # on the left of the array. This is done to reduce the
#    # search space for calculating the orbit loss.
#    return(grid[min_x0:min_x0+o_nx0, min_x1:min_x1+o_nx1].astype('uint8'))
#    
#def gen_rand_poly_images(n = 1000,img_size = 64,outp_size=48, n_point = 60, alpha = .2):
#    """
#    Generates a polygon by computing a randomly generated two
#    dimentional concave hull. The function returns a sample of
#    the polygons one-dimentional images, and their degrees of 
#    rotations.
#    """
#    #randomizes the seeds in each worker process
#    np.random.seed()
#    
#    #generate objects at the center of the image
#    a = np.random.uniform(img_size // 4, (img_size * 3) // 4, size=(n_point,2))
#    points = [Point(a[i]) for i in range(n_point)]
#    concave_hull, edge_points = alpha_shape(points, alpha=alpha)
#    
#    #if the result is a multipolygon, select the first polygon
#    t_poly = concave_hull.buffer(1)
#    if(type(t_poly) == geometry.multipolygon.MultiPolygon):
#        t_poly = t_poly[0]
#    
#    rotation_angles = np.hstack((np.array([0.],dtype='float16'), np.random.uniform(0,360, size = n-1).astype('float16')))
#    rotated_polygons = [rotate(t_poly,k) for k in rotation_angles]
#    one_d_images = np.array([generate_one_d_image(rotated_polygons[k], nx=img_size,ny=img_size) for k in range(n)])
#    #two_d_img = two_d_image(t_poly, nx=img_size,ny=img_size)
#    two_d_images = np.array([two_d_image(p,nx0=img_size
#                                         ,nx1=img_size
#                                         ,o_nx0=outp_size
#                                         ,o_nx1=outp_size )
#                             for p in [rotate(t_poly,k) for k in range(360)]])
#
#    return(rotation_angles, one_d_images, two_d_images)

def randWalk(d):
    """
    generates a d-dimensional random walk array 
    d: dimension
    """
    y = 0
    arr = np.zeros(d)
    for i in range(d):
        y += np.random.normal(scale=1)
        arr[i] = y
    arr = arr - np.mean(arr) #making it a zero-mean array
    return(arr)

def smooth(arr, windowLen):
    """
    smooths the array by convolutions. takes a moving average 
    of len 'windowLen'
    
    Arguments:
    arr: the array that would be smoothed
    windowLen: specifies the length of the convolution filter
               which determines the number of points that would
               be averaged to generate a point
    """
    box = np.ones(windowLen)/windowLen
    arrSmooth = np.convolve(arr, box, mode='same')
    return(arrSmooth)

def genSignal(d,numObs=1024, noise=2, smoothingWindowLen=1):
    """
    generates a d-dimensional random walk array, where |array| = d
    
    Arguments: 
      d: dimension of the array
      windowLen: length of the convolutions. Bigger values result in a
                 more smooth array.
    """
    np.random.seed()
    arr = smooth(randWalk(d),smoothingWindowLen)
    arr = arr * ((d / (np.sum(arr**2)))**.5)
    
    arrObservations = np.array([np.roll(arr,np.random.randint(d)) \
                                + np.random.normal(scale=noise,size=d) 
                                for i in range(numObs)])
    arr = np.array([np.roll(arr,i) for i in range(d)])
    return(arr,arrObservations)


def genSignalAndAllShifts(d, noise=2, smoothingWindowLen=1):
    """
    generates a d-dimensional random walk array, where |array| = d
    and returns a noisy observation in all possible shifts.
    Arguments: 
      d: dimension of the array
      windowLen: length of the convolutions. Bigger values result in a
                 more smooth array.
    """
    np.random.seed()
    arr = smooth(randWalk(d),smoothingWindowLen)
    arr = arr * ((d / (np.sum(arr**2)))**.5)
    
    arrWithNoise = arr + np.random.normal(scale=noise,size=d) 
    arrObservations = np.array([ np.roll(arrWithNoise,i) for i in range(d) ])
    arr = np.array([np.roll(arr,i) for i in range(d)])
    return(arr,arrObservations)