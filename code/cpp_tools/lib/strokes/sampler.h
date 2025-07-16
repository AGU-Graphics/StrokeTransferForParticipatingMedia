// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: sampler.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides a set of attribute samplers that given a (screen) location
// returns the attribute values (scalar, vectors).
//
// This file is part of the Stroke Transfer for Participating Media project.
// Released under the Creative Commons Attribution-NonCommercial (CC-BY-NC) license.
// See https://creativecommons.org/licenses/by-nc/4.0/ for details.
//
// DISCLAIMER:
// This code is provided "as is", without warranty of any kind, express or implied,
// including but not limited to the warranties of merchantability, fitness for a
// particular purpose, and noninfringement. In no event shall the authors or
// copyright holders be liable for any claim, damages or other liability.
// -----------------------------------------------------------------------------

#ifndef __sampler_h__
#define __sampler_h__

#include <Eigen/Core>

class Sampler
{
public:
	virtual ~Sampler(){};
	virtual void sample( const float in_x, const float in_y, float* io_values ) const = 0;
};

template <int N>
class ConstantSampler: public Sampler
{
public:
	ConstantSampler( const float in_const[N] )
	{
		for( int i=0; i<N; i++ )
			m_Const[i] = in_const[i];
	}
		
	void sample( const float in_x, const float in_y, float* io_values ) const
	{
		for( int i=0; i<N; i++ )
			io_values[i] = m_Const[i];		
	}
	
private:
	float m_Const[N];
	
	enum
	{
		NC = N
	};
};

template <int N>
class NearestSampler: public Sampler
{
	NearestSampler();
public:

	NearestSampler( const std::string& in_filename )
	{
		HDF5File h5( in_filename, HDF5AccessType::READ_ONLY );
		h5.readTensorAs1DArray( "", "data", m_Data, m_DataDim );
		if( m_DataDim.size() != N )
		{
			std::cout << "ERROR: NearestSampler< " << N << "> class can be only used for rank " << N << " tensors..." << std::endl;
			std::cout << "       The specified file <" << in_filename << "> requires a tensor in shape " << m_DataDim.transpose() << std::endl;
			std::cout << "       Exiting..." << std::endl;
			exit( -1 );
		}
		
		Eigen::VectorXf _resolution; Eigen::ArrayXi _dim;
		h5.readTensorAs1DArray( "", "resolution", _resolution, _dim );
		
		if( m_DataDim[1] != _resolution[0] || m_DataDim[0] != _resolution[1] )
		{
			std::cout << "ERROR: Mismatch in resolution data for NearestSampler<" << N << "> class: " << std::endl;
			std::cout << "       dim of data = " << m_DataDim[0] << "(rows) x " << m_DataDim[1] << "(cols)" << std::endl;
			std::cout << "       resolution = " << _resolution[1] << "(rows) x " << _resolution[0] << "(cols)" << std::endl;
		}
		
		m_Width = _resolution[0];
		m_Height = _resolution[1];
				
		m_Max_res = std::max<int>( m_Width, m_Height );
		m_Max_x = 1.0; 
		m_Max_y = 1.0; 
	}
	
	~NearestSampler()
	{}
		
	void sample( const float in_x, const float in_y, float* io_values ) const;
		
private:
	Eigen::VectorXf m_Data;
	Eigen::ArrayXi m_DataDim;
	int m_Width;
	int m_Height;
	float m_Max_x;
	float m_Max_y;
	int m_Max_res;
	
	enum
	{
		NC = N
	};
};

template<>
void NearestSampler<2>::sample( const float in_x, const float in_y, float* io_values ) const
{
	const int i = std::max<int>( 0, std::min<int>( m_Width-1, floor( m_Max_res * in_x / m_Max_x ) ) );
	const int j = std::max<int>( 0, std::min<int>( m_Height-1, floor( m_Max_res * in_y / m_Max_y ) ) );
	
	*io_values = m_Data[j*m_Width+i];
}

template<>
void NearestSampler<3>::sample( const float in_x, const float in_y, float* io_values ) const
{
	const int i = std::max<int>( 0, std::min<int>( m_Width-1, floor( m_Max_res * in_x / m_Max_x ) ) );
	const int j = std::max<int>( 0, std::min<int>( m_Height-1, floor( m_Max_res * in_y / m_Max_y ) ) );
	
	for( int k=0; k<m_DataDim[2]; k++ )
	{
		io_values[k] = m_Data[(j*m_Width+i)*m_DataDim[2]+k];
	}
}

class OrientationSampler: public Sampler
{
	OrientationSampler();
public:
	OrientationSampler( const std::string& in_filename )
	{
		HDF5File h5( in_filename, HDF5AccessType::READ_ONLY );
		h5.readTensorAs1DArray( "", "data", m_Data, m_DataDim );
		
		if( m_DataDim.size() != 3 || m_DataDim[2] != 2 )
		{
			std::cout << "ERROR: OrientationSampler class can be only used for rank 3 tensors with the third dimension being 2..." << std::endl;
			std::cout << "       The specified file <" << in_filename << "> requires a tensor in shape " << m_DataDim.transpose() << std::endl;
			std::cout << "       Exiting..." << std::endl;
			exit( -1 );
		}
		
		Eigen::VectorXf _resolution; Eigen::ArrayXi _dim;
		h5.readTensorAs1DArray( "", "resolution", _resolution, _dim );
		
		if( m_DataDim[1] != _resolution[0] || m_DataDim[0] != _resolution[1] )
		{
			std::cout << "ERROR: Mismatch in resolution data for OrientationSampler class: " << std::endl;
			std::cout << "       dim of data = " << m_DataDim[0] << "(rows) x " << m_DataDim[1] << "(cols)" << std::endl;
			std::cout << "       resolution = " << _resolution[1] << "(rows) x " << _resolution[0] << "(cols)" << std::endl;
		}
		
		m_Width = _resolution[0];
		m_Height = _resolution[1];
				
		m_Max_res = std::max<int>( m_Width, m_Height );
		m_Max_x = 1.0; 
		m_Max_y = 1.0; 
	}
	
	~OrientationSampler()
	{}
	
	void sample( const float in_x, const float in_y, float* io_values ) const
	{
		const float _if = m_Max_res * in_x / m_Max_x;
		const float _jf = m_Max_res * in_y / m_Max_y;
		
		const float s = _if - floor( _if );
		const float t = _jf - floor( _jf );
		
		const int i0 = std::max<int>( 0, std::min<int>( m_Width-1, floor( _if ) ) );
		const int j0 = std::max<int>( 0, std::min<int>( m_Height-1, floor( _jf ) ) );
		
		const int i1 = std::max<int>( 0, std::min<int>( m_Width-1, floor( _if ) + 1 ) );
		const int j1 = std::max<int>( 0, std::min<int>( m_Height-1, floor( _jf ) + 1 ) );
		
		io_values[0] = ( 1.0 - t ) * ( ( 1.0 - s ) * m_Data[(j0*m_Width+i0)*2+0] + s * m_Data[(j0*m_Width+i1)*2+0] ) 
			+ t * ( ( 1.0 - s ) * m_Data[(j1*m_Width+i0)*2+0] + s * m_Data[(j1*m_Width+i1)*2+0] );
		io_values[1] = ( 1.0 - t ) * ( ( 1.0 - s ) * m_Data[(j0*m_Width+i0)*2+1] + s * m_Data[(j0*m_Width+i1)*2+1] ) 
			+ t * ( ( 1.0 - s ) * m_Data[(j1*m_Width+i0)*2+1] + s * m_Data[(j1*m_Width+i1)*2+1] );
	}
	
private:
	Eigen::VectorXf m_Data;
	Eigen::ArrayXi m_DataDim;
	int m_Width;
	int m_Height;
	
	float m_Max_x;
	float m_Max_y;
	int m_Max_res;
};

class VelocitySampler: public Sampler
{
	VelocitySampler();
public:
	VelocitySampler( const std::string& in_vx_filename, const std::string& in_vy_filename )
	{
		HDF5File h5_vx( in_vx_filename, HDF5AccessType::READ_ONLY );
		Eigen::VectorXf _resolution_vx; Eigen::ArrayXi _dim_vx;
		h5_vx.readTensorAs1DArray( "", "resolution", _resolution_vx, _dim_vx );
		
		HDF5File h5_vy( in_vy_filename, HDF5AccessType::READ_ONLY );
		Eigen::VectorXf _resolution_vy; Eigen::ArrayXi _dim_vy;
		h5_vy.readTensorAs1DArray( "", "resolution", _resolution_vy, _dim_vy );
		
		if( _resolution_vx.size() != 2 || _resolution_vy.size() != 2 )
		{
			std::cout << "(VelocitySampler) resolution dimension error!!!" << std::endl;
			std::cout << "  _resolution_vx.size() and _resolution_vy.size() must be 2" << std::endl;
			std::cout << "  _resolution_vx.size(): " << _resolution_vx.size() << std::endl;
			std::cout << "  _resolution_vy.size(): " << _resolution_vy.size() << std::endl;
			exit(-1);
		}
		
		if( _resolution_vx(0) != _resolution_vy(0) || _resolution_vx(1) != _resolution_vy(1) )
		{
			std::cout << "(VelocitySampler) resolution mismatch!!!" << std::endl;
			std::cout << "  _resolution_vx must be equal to _resolution_vy" << std::endl;
			std::cout << "  _resolution_vx: " << _resolution_vx.transpose() << std::endl;
			std::cout << "  _resolution_vy: " << _resolution_vy.transpose() << std::endl;
			exit(-1);
		}
		
		m_Width = _resolution_vx(0);
		m_Height = _resolution_vx(1);
				
		h5_vx.readTensorAs1DArray( "", "data", m_Data_u, _dim_vx );
		h5_vy.readTensorAs1DArray( "", "data", m_Data_v, _dim_vy );
		
		if( m_Data_u.size() != m_Width*m_Height || m_Data_v.size() != m_Width*m_Height )
		{
			std::cout << "(VelocitySampler) data size mismatch!!!" << std::endl;
			std::cout << "  m_Data_u.size() and m_Data_v.size() must be equal to m_Width*m_Height" << std::endl;
			std::cout << "  m_Data_u.size(), m_Data_v.size(): " << m_Data_u.size() << ", " << m_Data_v.size() << std::endl;
			std::cout << "  m_Width*m_Height: " << m_Width*m_Height << std::endl;
			exit(-1);
		}
				
		m_Max_res = std::max<int>( m_Width, m_Height );
		m_Max_x = 1.0;
		m_Max_y = 1.0;
	}
	
	~VelocitySampler()
	{}
	
	void sample( const float in_x, const float in_y, float* io_values ) const
	{
		const float _if = m_Max_res * in_x / m_Max_x;
		const float _jf = m_Max_res * in_y / m_Max_y;
		
		const float s = _if - floor( _if );
		const float t = _jf - floor( _jf );
		
		const int i0 = std::max<int>( 0, std::min<int>( m_Width-1, floor( _if ) ) );
		const int j0 = std::max<int>( 0, std::min<int>( m_Height-1, floor( _jf ) ) );
		
		const int i1 = std::max<int>( 0, std::min<int>( m_Width-1, floor( _if ) + 1 ) );
		const int j1 = std::max<int>( 0, std::min<int>( m_Height-1, floor( _jf ) + 1 ) );
		
		io_values[0] = ( 1.0 - t ) * ( ( 1.0 - s ) * m_Data_u[j0*m_Width+i0] + s * m_Data_u[j0*m_Width+i1] ) 
			+ t * ( ( 1.0 - s ) * m_Data_u[j1*m_Width+i0] + s * m_Data_u[j1*m_Width+i1] );
		io_values[1] = ( 1.0 - t ) * ( ( 1.0 - s ) * m_Data_v[j0*m_Width+i0] + s * m_Data_v[j0*m_Width+i1] ) 
			+ t * ( ( 1.0 - s ) * m_Data_v[j1*m_Width+i0] + s * m_Data_v[j1*m_Width+i1] );
	}
	
private:
	Eigen::VectorXf m_Data_u;
	Eigen::VectorXf m_Data_v;
	
	int m_Width;
	int m_Height;
	
	float m_Max_x;
	float m_Max_y;
	int m_Max_res;
};

class RegionLabelSampler: public Sampler
{
	RegionLabelSampler();
public:

	RegionLabelSampler( const std::string& in_filename )
	{
		HDF5File h5( in_filename, HDF5AccessType::READ_ONLY );
		h5.readTensorAs1DArray( "", "cluster_labels", m_Labels, m_DataDim );
		
		if( m_DataDim.size() != 2 )
		{
			std::cout << "ERROR: RegionLabelSampler class can be only used for rank 2 tensors..." << std::endl;
			std::cout << "       The specified file <" << in_filename << "> requires a tensor in shape " << m_DataDim.transpose() << std::endl;
			std::cout << "       Exiting..." << std::endl;
			exit( -1 );
		}
		
		Eigen::VectorXf _resolution; Eigen::ArrayXi _dim;
		h5.readTensorAs1DArray( "", "resolution", _resolution, _dim );
		
		if( m_DataDim[1] != _resolution[0] || m_DataDim[0] != _resolution[1] )
		{
			std::cout << "ERROR: Mismatch in resolution data for RegionLabelSampler class: " << std::endl;
			std::cout << "       dim of data = " << m_DataDim[0] << "(rows) x " << m_DataDim[1] << "(cols)" << std::endl;
			std::cout << "       resolution = " << _resolution[1] << "(rows) x " << _resolution[0] << "(cols)" << std::endl;
		}
		
		m_Width = _resolution[0];
		m_Height = _resolution[1];

		m_Max_res = std::max<int>( m_Width, m_Height );
		m_Max_x = 1.0;
		m_Max_y = 1.0;
	}
	
	~RegionLabelSampler()
	{}
		
	void sample( const float in_x, const float in_y, float* io_values ) const
	{
		const int i = std::max<int>( 0, std::min<int>( m_Width-1, floor( m_Max_res * in_x / m_Max_x ) ) );
		const int j = std::max<int>( 0, std::min<int>( m_Height-1, floor( m_Max_res * in_y / m_Max_y ) ) );
	
		*io_values = m_Labels[j*m_Width+i];
	}
		
private:
	Eigen::VectorXi m_Labels;
	Eigen::ArrayXi m_DataDim;
	
	int m_Width;
	int m_Height;
	
	float m_Max_x;
	float m_Max_y;
	int m_Max_res;
};

#endif