import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    #print(channel_1_vals)
    #print(channel_2_vals)
    #print(channel_3_vals)
    
    # TODO: Compute histograms
    c1_hist = np.histogram(channel_1_vals, bins=32, range=(0,256))
    c2_hist = np.histogram(channel_2_vals, bins=32, range=(0,256))
    c3_hist = np.histogram(channel_3_vals, bins=32, range=(0,256))

    #print(c1_hist)
    #print(c2_hist)
    #print(c3_hist)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0])).astype(np.float64)
    #print(hist_features)
    normed_features = hist_features / np.sum(hist_features)
    #print(normed_features)
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    #print(min(norm_x_vals),max(norm_x_vals))
    #print(min(norm_y_vals),max(norm_y_vals))
    #print(min(norm_z_vals),max(norm_z_vals))

    # TODO: Compute histograms of normal values (just like with color)
    norm_x_hist = np.histogram(norm_x_vals, bins=32, range=(-1,2))
    norm_y_hist = np.histogram(norm_y_vals, bins=32, range=(-1,2))
    norm_z_hist = np.histogram(norm_z_vals, bins=32, range=(-1,2))

    # TODO: Concatenate and normalize the histograms

    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)
    #print(hist_features)
    normed_features = hist_features / np.sum(hist_features)
    #print(normed_features)
    return normed_features
