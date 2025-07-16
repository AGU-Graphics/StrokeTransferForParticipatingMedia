// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: activeset.cpp
// Maintainer: Yonghao Yue
//
// Description:
// This file implements hierarchical active set, a quad-tree style approach to 
// find active (unoccupied) pixel and deactivate the found pixel.
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

#include "activeset.h"
#include <lib/image/imageiopng.h>

HierarchicalActiveSet::HierarchicalActiveSet( const int in_width, const int in_height )
	: m_Width( in_width ), m_Height( in_height ), m_ActiveSet( nullptr )
{
	m_numLevelsW = 1;
	int _w = 1;
	for(;;)
	{
		if( _w >= m_Width )
			break;
		m_numLevelsW++;
		_w = _w << 1;
	}
	m_BaseLevelWidth = _w;
	
	m_numLevelsH = 1;
	int _h = 1;
	for(;;)
	{
		if( _h >= m_Height )
			break;
		m_numLevelsH++;
		_h = _h << 1;
	}
	m_BaseLevelHeight = _h;
	
	m_numLevels = std::min<int>( m_numLevelsW, m_numLevelsH );
	m_ActiveSet = new int*[ m_numLevels ];
	for( int i=0; i<m_numLevels; i++ )
	{
		m_ActiveSet[i] = new int[ _w * _h ];
		
		if( i == m_numLevels - 1 )
		{
			m_TopLevelWidth = _w;
			m_TopLevelHeight = _h;
		}
		
		_w = _w >> 1;
		_h = _h >> 1;
	}	
}

HierarchicalActiveSet::~HierarchicalActiveSet()
{
	if( m_ActiveSet != nullptr )
	{
		for( int i=0; i<m_numLevels; i++ )
		{
			delete[] m_ActiveSet[i];
		}
		delete[] m_ActiveSet;
		m_ActiveSet = nullptr;
	}
}

void HierarchicalActiveSet::setActiveSetFromRGBABuffer( const Image<float, 4>& in_buffer, const float threshold )
{
	// Base level
	for( int j=0; j<in_buffer.getHeight(); j++ )
	{
		for( int i=0; i<in_buffer.getWidth(); i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = in_buffer.getValue( i, j, 3 ) >= threshold ? 0 : 1;
		}
		for( int i=in_buffer.getWidth(); i<m_BaseLevelWidth; i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	for( int j=in_buffer.getHeight(); j<m_BaseLevelHeight; j++ )
	{
		for( int i=0; i<m_BaseLevelWidth; i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	
	// Higher levels
	int _w = m_BaseLevelWidth >> 1;
	int _h = m_BaseLevelHeight >> 1;
	
	for( int k=0; k<m_numLevels-1; k++ )
	{
		for( int j=0; j<_h; j++ )
		{
			for( int i=0; i<_w; i++ )
			{
				m_ActiveSet[k+1][j*_w+i] = m_ActiveSet[k][(2*j)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j)*(_w*2)+(2*i+1)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i+1)];
			}
		}
		
		_w = _w >> 1;
		_h = _h >> 1;
	}
}

void HierarchicalActiveSet::setActiveSetFromRGBABufferBytes( const unsigned char* in_buffer, const int in_buffer_width, const int in_buffer_height, const float _threshold )
{
	const unsigned char threshold = (unsigned char)( _threshold * 255 );
	// Base level
	for( int j=0; j<in_buffer_height; j++ )
	{
		for( int i=0; i<in_buffer_width; i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = in_buffer[(j*in_buffer_width+i)*4+3] >= threshold ? 0 : 1;
		}
		for( int i=in_buffer_width; i<m_BaseLevelWidth; i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	for( int j=in_buffer_height; j<m_BaseLevelHeight; j++ )
	{
		for( int i=0; i<m_BaseLevelWidth; i++ )
		{
			m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	
	// Higher levels
	int _w = m_BaseLevelWidth >> 1;
	int _h = m_BaseLevelHeight >> 1;
	
	for( int k=0; k<m_numLevels-1; k++ )
	{
		for( int j=0; j<_h; j++ )
		{
			for( int i=0; i<_w; i++ )
			{
				m_ActiveSet[k+1][j*_w+i] = m_ActiveSet[k][(2*j)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j)*(_w*2)+(2*i+1)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i+1)];
			}
		}
		
		_w = _w >> 1;
		_h = _h >> 1;
	}
}

void HierarchicalActiveSet::updateActiveSetFromRGBABuffer( const Image<float, 4>& in_buffer, int in_x, int in_y, int in_w, int in_h, const float threshold )
{
	int l = in_x;
	int t = in_y;
	int r = in_x + in_w - 1;
	int b = in_y + in_h - 1;
	
	//Base level
	for( int j=t; j<=b; j++ )
	{
		for( int i=l; i<=r; i++ )
		{
			if( in_buffer.getValue( i, j, 3 ) >= threshold )
				m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	
	//Higher levels
	int _w = m_BaseLevelWidth >> 1;
	for( int k=0; k<m_numLevels-1; k++ )
	{
		l = l >> 1;
		t = t >> 1;
		r = r >> 1;
		b = b >> 1;
		
		for( int j=t; j<=b; j++ )
		{
			for( int i=l; i<=r; i++ )
			{
				m_ActiveSet[k+1][j*_w+i] = m_ActiveSet[k][(2*j)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j)*(_w*2)+(2*i+1)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i+1)];
			}
		}
		_w = _w >> 1;
	}
}

void HierarchicalActiveSet::updateActiveSetFromRGBABufferBytes( const unsigned char* in_buffer, const int in_buffer_width, int in_x, int in_y, int in_w, int in_h, const float _threshold )
{
	const unsigned char threshold = (unsigned char)( _threshold * 255 );
	
	int l = in_x;
	int t = in_y;
	int r = in_x + in_w - 1;
	int b = in_y + in_h - 1;
	
	//Base level
	for( int j=t; j<=b; j++ )
	{
		for( int i=l; i<=r; i++ )
		{
			if( in_buffer[(j*in_buffer_width+i)*4+3] >= threshold )
				m_ActiveSet[0][j*m_BaseLevelWidth+i] = 0;
		}
	}
	
	//Higher levels
	int _w = m_BaseLevelWidth >> 1;
	for( int k=0; k<m_numLevels-1; k++ )
	{
		l = l >> 1;
		t = t >> 1;
		r = r >> 1;
		b = b >> 1;
		
		for( int j=t; j<=b; j++ )
		{
			for( int i=l; i<=r; i++ )
			{
				m_ActiveSet[k+1][j*_w+i] = m_ActiveSet[k][(2*j)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j)*(_w*2)+(2*i+1)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i)] + m_ActiveSet[k][(2*j+1)*(_w*2)+(2*i+1)];
			}
		}
		_w = _w >> 1;
	}
}

int HierarchicalActiveSet::findUnoccupied( const float in_random_number, int& out_x, int& out_y )
{
	int sum = 0;
	for( int j=0; j<m_TopLevelHeight; j++ )
	{
		for( int i=0; i<m_TopLevelWidth; i++ )
		{
			sum += m_ActiveSet[m_numLevels-1][j*m_TopLevelWidth+i];
		}
	}
	
	if( sum <= 0 )
		return sum;
	
	int target = std::min<int>( sum - 1, std::max<int>( 0, floor( in_random_number * sum ) ) );
	int _t = target;
	
	//Top level
	int _w = m_TopLevelWidth;
	bool found = false;
	for( out_y=0; out_y<m_TopLevelHeight; out_y++ )
	{
		for( out_x=0; out_x<m_TopLevelWidth; out_x++ )
		{
			if( _t >= m_ActiveSet[m_numLevels-1][out_y*_w+out_x] )
				_t -= m_ActiveSet[m_numLevels-1][out_y*_w+out_x];
			else
			{
				found = true;
				break;
			}
		}
		if( found )
			break;
	}
	
	if( !found )
		std::cout << "ERROR: NOT FOUND!!!" << std::endl;
	
	//Lower levels
	for( int k=0; k<m_numLevels-1; k++ )
	{
		bool found = false;
		_w = _w << 1;
		
		int j=0, i=0;
		for( j=0; j<2; j++ )
		{
			for( i=0; i<2; i++ )
			{
				if( _t >= m_ActiveSet[m_numLevels-2-k][(2*out_y+j)*_w+(2*out_x+i)] )
					_t -= m_ActiveSet[m_numLevels-2-k][(2*out_y+j)*_w+(2*out_x+i)];
				else
				{
					found = true;
					break;
				}
			}
			if( found )
				break;
		}
		
		if( !found )
			std::cout << "ERROR: NOT FOUND!!! at level " << m_numLevels-2-k << std::endl;
		
		out_x = 2*out_x + i;
		out_y = 2*out_y + j;
	}
	
	return sum;
}

void HierarchicalActiveSet::save_active_set_as_images( const std::string& in_filename_template )
{
	Image<float, 1> image;
	int _w = m_BaseLevelWidth;
	int _h = m_BaseLevelHeight;
	float scale = 1.0;
	
	for( int k=0; k<m_numLevels; k++ )
	{
		image.init( _w, _h );
		for( int j=0; j<_h; j++ )
		{
			for( int i=0; i<_w; i++ )
			{
				image.setValue( i, j, 0, m_ActiveSet[k][j*_w+i] / scale );
			}
		}
		
		char fn[256];
		snprintf( fn, 256, in_filename_template.c_str(), k );
		saveImagePng( fn, image );
		
		_w = _w >> 1;
		_h = _h >> 1;
		scale *= 4.0;
	}
}

void HierarchicalActiveSet::viewValues( const int in_x, const int in_y )
{
	std::cout << "viewValues() called for " << in_x << ", " << in_y << std::endl;
	int _w = m_BaseLevelWidth;
	int x = in_x;
	int y = in_y;
	for( int k=0; k<m_numLevels; k++ )
	{
		std::cout << "k: " << k << ", " << m_ActiveSet[k][y * _w + x] << std::endl;
		y = y >> 1;
		x = x >> 1;
	}
}

bool HierarchicalActiveSet::checkSummationConsistency()
{
	int _w = m_BaseLevelWidth;
	int _h = m_BaseLevelHeight;
	
	int sum = 0;
	for( int j=0; j<_h; j++ )
	{
		for( int i=0; i<_w; i++ )
		{
			sum += m_ActiveSet[0][j*_w+i];
		}
	}
	
	for( int k=0; k<m_numLevels-1; k++ )
	{
		_w = _w >> 1;
		_h = _h >> 1;
		int _sum = 0;
		for( int j=0; j<_h; j++ )
		{
			for( int i=0; i<_w; i++ )
			{
				_sum += m_ActiveSet[k+1][j*_w+i];
			}
		}
		if( _sum != sum )
		{
			return false;
		}
	}
	
	return true;
}

void HierarchicalActiveSet::showNonZeros()
{
	int _w = m_BaseLevelWidth;
	int _h = m_BaseLevelHeight;
	for( int k=0; k<m_numLevels; k++ )
	{
		std::cout << "Level " << k << ": ";
		for( int j=0; j<_h; j++ )
		{
			for( int i=0; i<_w; i++ )
			{
				if( m_ActiveSet[k][j*_w+i] != 0 )
					std::cout << "(" << i << "," << j << "); ";
			}
		}
		std::cout << std::endl;
		_w = _w >> 1;
		_h = _h >> 1;
	}
}
