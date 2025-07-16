# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: silhouette_distance.py
# Maintainer: Naoto Shirashima
#
# Description:
# Generate silhouette distance field from transmittance
#
# This file is part of the Stroke Transfer for Participating Media project.
# Released under the Creative Commons Attribution-NonCommercial (CC-BY-NC) license.
# See https://creativecommons.org/licenses/by-nc/4.0/ for details.
#
# DISCLAIMER:
# This code is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose, and noninfringement. In no event shall the authors or
# copyright holders be liable for any claim, damages or other liability.
# -----------------------------------------------------------------------------
import numpy as np
import sys
import h5py
import os
import cv2
import argparse
from scipy.special import gamma
from data_io import DataIO2D, Output2D
import math

def p( t, th_t, sigma ):
    return np.exp( - ( ( t - th_t ) ** 2 ) / ( 2.0 * sigma ** 2 ) ) / np.sqrt( 2.0 * np.pi * sigma ** 2 )
    
def compute_sample_points( max_inv_derivs, epsilon, _res, th_t, sigma ):
    th = ( np.arange( _res + 1 ) ) / _res
    _w = p( th, th_t, sigma )
    __w = 0.5 * ( _w[1:] + _w[0:-1] )
    w = __w / ( np.sum(__w) * ( 1.0 / _res ) )
    
    F = np.cumsum( np.insert(w, 0, 0.0) ) / _res
    G = np.cumsum( F ) / _res
    Q_1 = 0.5 * ( th[1:] - th[0:-1] )
    Q_2 = F[1:] + F[0:-1]
    Q_3 = G[1:] - G[0:-1]
    
    Q = np.abs( Q_1 * Q_2 - Q_3 ) 
    
    eta = []
    li = []
    
    t = epsilon / _res
    
    accum = 0.0
    l = 0.0
    seg_start = 0.0
    
    for i in range( _res ):
        c = ( th[i] + th[i+1] ) * 0.5
        idx_max_inv_derivs = min( [ len( max_inv_derivs ) - 1 , max( [ 0, int( math.floor( c * len( max_inv_derivs ) ) ) ] ) ] )
            
        accum += Q[i] * max_inv_derivs[idx_max_inv_derivs]
        l += th[i+1] - th[i]
        
        if accum >= t:
            eta.append( 0.5 * ( th[i+1] + seg_start ) )
            li.append(l)
            l = 0.0
            accum = 0.0
            seg_start = th[i+1]

    if len( eta ) == 0 or eta[-1] != th[-1]:
        eta.append( th[-1] )
        li.append( l )
                    
    return np.array( eta ), np.array( li )


def silhouette_distance_field_2(intput_data, g_th_t, g_sigma):
    padding_pixel = 1
    padding = padding_pixel > 0
    res = 500
    th = (0.5 + np.arange(res)) / res
    w = p(th, g_th_t, g_sigma)
    w = w / np.sum(w)

    distance = np.zeros_like(intput_data)
    distance_plus = np.zeros_like(intput_data)
    distance_minus = np.zeros_like(intput_data)

    if padding:
        input_data_padding = np.zeros((intput_data.shape[0] + padding_pixel*2, intput_data.shape[1] + padding_pixel*2))
        input_data_padding[padding_pixel:-padding_pixel, padding_pixel:-padding_pixel] = intput_data
        input_data_padding[0, :] = input_data_padding[-1, :] = input_data_padding[:, 0] = input_data_padding[:, -1] = 1
    else:
        input_data_padding = intput_data

    transmittance_bin_plus = np.zeros_like(input_data_padding)

    for i in range(res):
        t, transmittance_bin = cv2.threshold(input_data_padding, th[i], 255, cv2.THRESH_BINARY)

        _transmittance_bin_plus = np.uint8(transmittance_bin) - 255
        _transmittance_bin_minus = np.uint8(transmittance_bin)

        tmp = transmittance_bin - np.ones_like(transmittance_bin) * transmittance_bin[0, 0]
        # print(np.all(tmp == 0.0))
        if np.all(tmp == 0.0):
            continue

        transmittance_bin_plus += _transmittance_bin_plus * w[i]
        distance_plus_padding = cv2.distanceTransform(_transmittance_bin_plus, cv2.DIST_L2, 0)
        distance_minus_padding = cv2.distanceTransform(_transmittance_bin_minus, cv2.DIST_L2, 0)
        if padding:
            _distance_plus = distance_plus_padding[padding_pixel:-padding_pixel, padding_pixel:-padding_pixel]
            _distance_minus = distance_minus_padding[padding_pixel:-padding_pixel, padding_pixel:-padding_pixel]
        else:
            _distance_plus = distance_plus_padding
            _distance_minus = distance_minus_padding

        width = intput_data.shape[1]

        _distance_plus /= width
        _distance_minus /= width

        _distance = distance_plus - distance_minus

        distance += _distance * w[i]
        distance_plus += _distance_plus * w[i]
        distance_minus += _distance_minus * w[i]

    return distance, distance_plus, distance_minus

def silhouette_distance_field_3( input_transmittance_field, transmittance_representative_threshold, sigma ):
    padding_pixel = 1
    padding = padding_pixel > 0
    
    max_res = max( [ input_transmittance_field.shape[0], input_transmittance_field.shape[1] ] )
    dx = 1.0 / max_res
    deriv_y, deriv_x = np.gradient( input_transmittance_field, dx, dx )
    deriv = np.sqrt( deriv_x * deriv_x + deriv_y * deriv_y )
    
    slice_res = 200
    eps = 1.0e-3
    slices = np.arange( slice_res + 1 ) / slice_res
    max_inv_derivs = []
    for i in range( slice_res ):
        cond = ( input_transmittance_field > slices[i] ) & ( input_transmittance_field <= slices[i+1] )
        pixel_idx_within_slice = np.asarray( cond ).nonzero()
        if len( pixel_idx_within_slice[0] ) == 0:
            max_inv_derivs.append( eps )
        else:
            max_inv_derivs.append( np.max( 1.0 / ( deriv[ pixel_idx_within_slice ] + eps ) ) )
    
    xi, li = compute_sample_points( max_inv_derivs, 0.01, 20000, transmittance_representative_threshold, sigma )
    w = p( xi, transmittance_representative_threshold, sigma ) * np.array( li )
    w = w / np.sum(w)

    distance = np.zeros_like( input_transmittance_field )
    distance_plus = np.zeros_like( input_transmittance_field )
    distance_minus = np.zeros_like( input_transmittance_field )

    if padding:
        input_data_padding = np.zeros( ( input_transmittance_field.shape[0] + padding_pixel*2, input_transmittance_field.shape[1] + padding_pixel*2 ) )
        input_data_padding[ padding_pixel:-padding_pixel, padding_pixel:-padding_pixel ] = input_transmittance_field
        input_data_padding[0, :] = input_data_padding[-1, :] = input_data_padding[:, 0] = input_data_padding[:, -1] = 1
    else:
        input_data_padding = input_transmittance_field

    transmittance_bin_plus = np.zeros_like( input_data_padding )

    for i in range( w.size ):
        t, transmittance_bin = cv2.threshold( input_data_padding, xi[i], 255, cv2.THRESH_BINARY )

        _transmittance_bin_plus = np.uint8( transmittance_bin ) - 255
        _transmittance_bin_minus = np.uint8( transmittance_bin )

        tmp = transmittance_bin - np.ones_like( transmittance_bin ) * transmittance_bin[0, 0]
        if np.all(tmp == 0.0):
            continue

        transmittance_bin_plus += _transmittance_bin_plus * w[i]
        distance_plus_padding = cv2.distanceTransform( _transmittance_bin_plus, cv2.DIST_L2, 0 )
        distance_minus_padding = cv2.distanceTransform( _transmittance_bin_minus, cv2.DIST_L2, 0 )
        if padding:
            _distance_plus = distance_plus_padding[padding_pixel:-padding_pixel, padding_pixel:-padding_pixel]
            _distance_minus = distance_minus_padding[padding_pixel:-padding_pixel, padding_pixel:-padding_pixel]
        else:
            _distance_plus = distance_plus_padding
            _distance_minus = distance_minus_padding

        width = input_transmittance_field.shape[1]

        _distance_plus /= width
        _distance_minus /= width

        _distance = distance_plus - distance_minus

        distance += _distance * w[i]
        distance_plus += _distance_plus * w[i]
        distance_minus += _distance_minus * w[i]

    return distance, distance_plus, distance_minus, transmittance_bin_plus


def GetSilhouetteFeature(input_data, threshold_value, sigma):
    try:
        distance_field_plus_minus_3, distance_field_plus_3, distance_field_minus_3, _ = silhouette_distance_field_3(input_data, threshold_value, sigma)
        # distance_field_plus_minus_3, distance_field_plus_3, distance_field_minus_3 = silhouette_distance_field_2( input_data, threshold_value, sigma )
    except Exception as e:
        raise e
    
    return distance_field_plus_minus_3


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, help='input hdf5 file', required=True)
    args.add_argument('-t', type=float, help='threshold value', required=True)
    args.add_argument('-o', type=str, help='output hdf5 file', required=True)
    args.add_argument('--extra_resolution', type=float, default=0, help='extra resolution')
    args = args.parse_args()
    input_hdf5_file = args.i
    threshold_value = args.t
    output_hdf5_file = args.o
    extra_resolution = args.extra_resolution
    sigma = 0.2

    h5 = DataIO2D.InitLoadFile( input_hdf5_file, taichi=False )
    data = h5.data_np

    out_data = GetSilhouetteFeature(data, threshold_value, sigma)
    out_data = np.array(out_data)
    
    if extra_resolution > 0:
        tmp_res = np.array(out_data.shape) / extra_resolution / 2
        center_res = np.array(out_data.shape) / 2
        start_index = center_res - tmp_res
        end_index = center_res + tmp_res
        start_index = start_index.astype(int)
        end_index = end_index.astype(int)
        tmp_data = out_data[start_index[0]:end_index[0], start_index[1]:end_index[1]]
        out_data = tmp_data

    Output2D( out_data, output_hdf5_file )


if __name__ == '__main__':
    main()





v1="""
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import cv2

def silhouette_distance_field(intput_data, th):
    t, transmittance_bin = cv2.threshold(intput_data, th, 255, cv2.THRESH_BINARY)
    transmittance_bin_plus = np.uint8(transmittance_bin) - 255
    transmittance_bin_minus = np.uint8(transmittance_bin)

    distance_plus = cv2.distanceTransform(transmittance_bin_plus, cv2.DIST_L2, 0)
    distance_minus = cv2.distanceTransform(transmittance_bin_minus, cv2.DIST_L2, 0)

    w = intput_data.shape[1]

    distance_plus /= w
    distance_minus /= w

    distance = distance_plus - distance_minus

    return distance, distance_plus, distance_minus
    

def GetSilhouetteFeature(input_data, threshold_value):
    is_standardizationed_hdf5 = False
    result_plus_minus = []
    
    distance_field_plus_minus, distance_field_plus, distance_field_minus = silhouette_distance_field(input_data, threshold_value)
    if False:
        plt.imshow(distance_field)
        plt.colorbar()
        plt.contour(transmittance, [threshold_value], colors="red")
        plt.show()
    return distance_field_plus_minus

def main():
    input_hdf5_file = sys.argv[1]
    threshold_value = float(sys.argv[2])
    output_hdf5_file = sys.argv[3]

    target_resolution=(512,512)

    input_h5 = h5py.File(input_hdf5_file, 'r')
    resolution = np.array(input_h5['resolution']).flatten()
    print('resolution', resolution)
    data = np.array(input_h5['data'])
    data = data.reshape([resolution[1], resolution[0]])
    
    print('data.shape', data.shape)
    out_data = GetSilhouetteFeature(data, threshold_value)
    out_data = np.array(out_data)
    print('out_data.shape', out_data.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(data, origin='lower')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(out_data, origin='lower')
    plt.colorbar()
    plt.savefig(f'{output_hdf5_file}.png')
    
    
    print('data.shape', data.shape)
    output_h5 = h5py.File(f'{output_hdf5_file}.hdf5', 'w')
    output_h5.create_dataset('data', data=np.array(out_data.flatten(), dtype=np.float32))
    output_h5.create_dataset('resolution', data=resolution)
    # output_h5.create_dataset('bb_min', data=input_h5['bb_min'])
    # output_h5.create_dataset('bb_max', data=input_h5['bb_max'])
    # output_h5.create_dataset('cell_size', data=input_h5['cell_size'])
    output_h5.close()

if __name__ == '__main__':
    main()
"""