# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: data_io.py
# Maintainer: Naoto Shirashima
#
# Description:
# 3D, 2D data input/output
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
import h5py
import taichi as ti
import taichi.math as tm
import time

@ti.data_oriented
class DataIO3D:
    def __init__( self, checkSumTest=True ):
        self.bb_min = None
        self.cell_width = None
        self.data_np = None
        self.data = None
        if checkSumTest:
            self.taichi_sum = ti.field(dtype=ti.f32, shape=())
        
    @ti.kernel
    def compute_taichi_check_sum( self, data: ti.template() ):
        self.taichi_sum[None] = 0.0
        for i, j, k in data:
            self.taichi_sum[None] += data[i, j, k]
        
    @classmethod
    def InitLoadFile( cls, input_file, taichi=True, checkSumTest=True ):
        checkSumTest = False
        self = DataIO3D( checkSumTest=checkSumTest )
        h5 = h5py.File( input_file, 'r' )
        self.bb_min = tm.vec3( h5[ 'bb_min' ][()] )
        tmp_cell_width = np.array( h5['cell_size'][()] )
        if tmp_cell_width.ndim == 0:
            self.cell_width = tm.vec3( [ tmp_cell_width, tmp_cell_width, tmp_cell_width ] )
        elif tmp_cell_width.ndim == 1:
            self.cell_width = tm.vec3([h5['cell_size'][()][0], h5['cell_size'][()][0], h5['cell_size'][()][0]])
        else:
            print( 'tmp_cell_width.ndim', tmp_cell_width.ndim )
            print( 'cell width ndim error' )
            exit( -1 )
        resolution = np.array( h5['resolution'] ).flatten()
        self.data_np = np.array( h5['data'][:]).reshape( [ resolution[0], resolution[1], resolution[2] ], order='F' )
        if taichi == True:
            self.data = DataIO3D.CPU2Taichi( self.data_np )
        h5.close()

        if taichi and checkSumTest:
            self.compute_taichi_check_sum( self.data )
            taichi_sum_result = self.taichi_sum.to_numpy()
            numpy_sum_result = np.sum( self.data_np )
            
            relative_diff = abs( numpy_sum_result - taichi_sum_result ) / max( abs(numpy_sum_result), abs(taichi_sum_result) )
            print( '+' )
            print( 'sum(data_np)', numpy_sum_result )
            print( 'taichi_sum(data)', taichi_sum_result )
            print( 'relative_diff', relative_diff )
            
            if numpy_sum_result == 0 and taichi_sum_result == 0:
                print( 'ERROR!!! sum(data_np) == 0 && taichi_sum(data) == 0' )
                print( 'Could not read data from <' + input_file + '>' )
                exit( -1 )
            # elif abs( numpy_sum_result - taichi_sum_result ) < 1.0e-4:
            elif relative_diff < 0.01:
                print( 'Check sum test passed.' )
            else:
                print( 'Data mismatch found between taichi data and numpy data. Two possible causes are 1) the line integration did not complete successfully, or 2) memory allocation failed within taichi' )
                exit( -1 )
                
        return self
        
    @classmethod
    def InitTemplate( cls, bb_min, cell_width, resolution, data: ti.template() ):
        self = DataIO3D()
        self.bb_min = bb_min
        self.cell_width = cell_width
        self.data = data
        return self
    
    @classmethod
    def TaichiFieldInit( cls, data_shape, data_type=ti.f32 ):
        data = ti.field( dtype=data_type, shape=( data_shape[0], data_shape[1], data_shape[2] ) )
        return data
    
    @classmethod
    def CPU2Taichi( cls, np ):
        res = np.shape
        data = DataIO3D.TaichiFieldInit( res )
        data.from_numpy( np )
        return data

    def bb_max_python( self ):
        return self.bb_min + self.cell_width * np.array([self.data_np.shape[0]-1, self.data_np.shape[1]-1, self.data_np.shape[2]-1])
    def bb_max_taichi(self) -> tm.vec3:
        return tm.vec3(self.bb_max_python())
    @ti.func
    def bb_max_taichi_func( self ) -> tm.vec3:
        return tm.vec3(self.bb_min + self.cell_width * tm.vec3([self.data_np.shape[0]-1, self.data_np.shape[1]-1, self.data_np.shape[2]-1]))
    def get_shape(self):
        return self.data_np.shape


def Output3D( bb_min, cell_width, data_np, output_file, compress=True ):
    if output_file is None or output_file == '':
        raise ValueError("Output file path cannot be None or empty.")
    h5 = h5py.File( output_file, 'w' )
    h5.create_dataset( 'bb_min', data=np.array( bb_min, dtype=np.float32 ) )
    h5.create_dataset( 'cell_size', data=np.array( cell_width, dtype=np.float32 ) )
    res = np.array( data_np.shape )
    h5.create_dataset( 'resolution', data=np.array( res, dtype=np.int32 ) )
    if compress:
        h5.create_dataset( 'data', data=data_np.flatten( order='F' ), dtype=np.float32, compression='gzip' )
    else:
        h5.create_dataset( 'data', data=data_np.flatten( order='F' ), dtype=np.float32 )
    h5.close()

class DataIO2D:
    screen_bb_min_0 = 0.0
    screen_bb_min_1 = 0.0
    screen_cell_width = 1.0
    def __init__(self):
        self.bb_min = None
        self.cell_width = None
        self.bb_max = None
        self.data_np = None
        self.data = None
        
    @classmethod
    def InitLoadFile(cls,input_file_path, taichi=True):
        self = DataIO2D()
        h5 = h5py.File(input_file_path, 'r')
        resolution = np.array(h5['resolution']).flatten()
        old_res = np.array(resolution)
        resolution[0] = old_res[1]
        resolution[1] = old_res[0]
        self.data_np = np.array(h5['data'][:]).reshape( resolution, order='C')
        self.data_np = np.squeeze(self.data_np)
        resolution = self.data_np.shape
        if taichi == True:
            self.data = ti.field(dtype=ti.f32, shape=( resolution ))
            self.data.from_numpy(self.data_np)
        h5.close()

        self.bb_min = tm.vec2([self.screen_bb_min_0, self.screen_bb_min_1])
        self.cell_width = self.screen_cell_width
        self.bb_max = self.bb_max_taichi()
        return self
        
    @classmethod
    def InitTemplate(cls, bb_min, cell_width, resolution, data: ti.template()):
        self = DataIO2D()
        self.bb_min = bb_min
        self.cell_width = cell_width
        self.data = data
        return self
    
    @classmethod
    def TaichiFieldInit(cls, shape, dtype=ti.f32):
        data = ti.field(dtype=dtype, shape=(shape))
        return data

    def get_shape(self):
        return self.data_np.shape
    def bb_max_python(self):
        return self.bb_min + self.cell_width * np.array([self.data_np.shape[1], self.data_np.shape[0]])
    def bb_max_taichi(self) -> tm.vec2:
        return tm.vec2(self.bb_max_python())

def Output2D(data_np, output_file_path):
    if output_file_path is None or output_file_path == '':
        raise ValueError("Output file path cannot be None or empty.")
    h5 = h5py.File(output_file_path, 'w')
    old_res = data_np.shape
    resolution = np.array(old_res)
    resolution[0] = old_res[1]
    resolution[1] = old_res[0]
    h5.create_dataset('resolution', data=resolution, dtype=np.int32)
    h5.create_dataset('data', data=data_np.flatten(order='C'), dtype=np.float32)
    h5.close()
