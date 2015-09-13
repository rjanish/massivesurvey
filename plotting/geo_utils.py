#! /usr/bin/python

import scipy as sp
from scipy.special import cosdg, sindg
import shapely.geometry as geo

def rotate_point(coords, angle):
    # rotate coords by angle counterclockwise
    rot_angle = float(angle) % 360
    rotation_matrix = sp.matrix([[cosdg(rot_angle), -sindg(rot_angle)],
                                 [sindg(rot_angle), cosdg(rot_angle)]])
    point_vector = sp.transpose(sp.matrix(coords))
    rotated_point_vector = rotation_matrix*point_vector
    x_rotated, y_rotated = sp.array(sp.transpose(rotated_point_vector))[0]
    return x_rotated, y_rotated

def string_to_tuple(string):
    # string must have format: "(numbertype,numbertype)"
    return tuple([float(n) for n in string.strip("()").split(",")])

def approxArc(center, radius, thmin, thmax, resolution=24):
    # returns a shapely LineString approximating an arc
    total_delta_angle = thmax - thmin
    number_of_samples = int(sp.ceil(resolution*4*abs(total_delta_angle)/360))+1
    sample_angles = sp.linspace(thmin, thmax, number_of_samples)
    x = center[0] + radius*cosdg(sample_angles)
    y = center[1] + radius*sindg(sample_angles)
    sample_points = [(x[n], y[n]) for n in range(number_of_samples)]
    return geo.LineString(sample_points)

def polar_box(rmin, rmax, thmin, thmax):
    # returns a polygon with boundaries given by constant polar coordinates
    delta_theta = (thmax - thmin) % 360
    thmax = thmin + delta_theta
    maxarc_coords = list(approxArc((0.0, 0.0), rmax, thmin, thmax).coords)
    minarc_coords = list(approxArc((0.0, 0.0), rmin, thmin, thmax).coords)
    minarc_coords.reverse()
    return geo.Polygon(maxarc_coords + minarc_coords)

def plot_shapely(ax, shapely_geometry, color='b',
               line_style='-', point_style='o', border=1.0):
    # plot the outline of a shapely Polygon, LineString, or Point to ax
    if shapely_geometry.area > 0.0:
        shapely_geometry = shapely_geometry.boundary
    try:
        num_shapes = len(shapely_geometry)
    except:
        num_shapes = 1
        shapely_geometry = [shapely_geometry]
    for single_geometry in shapely_geometry:
        verticies = sp.array(single_geometry.coords)
        if len(verticies) > 1:
            style = color + line_style
        elif len(verticies) == 1:
            style = color + point_style
        else:
            raise Exception("shapley object must contain at least 1 point")
        ax.plot(verticies[:,0], verticies[:,1], style)
        current_limits = list(ax.axis())
        x_coords = sp.concatenate((verticies[:,0] + border,
                                   verticies[:,0] - border,
                                   current_limits[0:2]))
        y_coords = sp.concatenate((verticies[:,1] + border,
                                   verticies[:,1] - border,
                                   current_limits[2:4]))
        ax.axis([min(x_coords), max(x_coords),
                 min(y_coords), max(y_coords)])
    return
