// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: image.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides a class for (2D) images.
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

#ifndef image_h
#define image_h

#include <stdint.h>
#include <iostream>

template<typename T, int N>
class Image
{
public:
	Image()
		: m_Width(0), m_Height(0), m_Ptr(nullptr)
	{}

	~Image()
	{
		finalize();
	}

	inline bool init( const int32_t in_Width, const int32_t in_Height )
	{
		if( ( in_Width <= 0 ) || ( in_Height <= 0 ) )
		{
			m_Width = 0;
			m_Height = 0;
			m_Ptr = NULL;
		}
		else
		{
			m_Width = in_Width;
			m_Height = in_Height;
			m_Ptr = (T*) malloc( sizeof(T) * in_Width * in_Height * N );
		}

		return true;
	}

	inline bool clear()
	{
		memset( m_Ptr, 0, sizeof(T) * m_Width * m_Height * N );
		return true;
	}

	inline bool resize( const int32_t in_Width, const int32_t in_Height )
	{
		if( ( in_Width <= 0 ) || ( in_Height <= 0 ) )
		{}
		else if( ( in_Width == m_Width ) || ( in_Height == m_Height ) )
		{}
		else
		{
			m_Width = in_Width;
			m_Height = in_Height;
			m_Ptr = (T*) realloc( m_Ptr, sizeof(T) * in_Width * in_Height * N );
		}

		return true;
	}

	inline bool finalize()
	{
		m_Width = 0;
		m_Height = 0;
		if( m_Ptr != NULL )
		{
			free( m_Ptr );
			m_Ptr = NULL;
		}

		return true;
	}

	inline void setValue( const int32_t i, const int32_t j, const int32_t k, const T value )
	{
		m_Ptr[ ( j * m_Width + i ) * N + k ] = value;
	}

	inline void addValue( const int32_t i, const int32_t j, const int32_t k, const T value )
	{
		m_Ptr[ ( j * m_Width + i ) * N + k ] += value;
	}

	inline T getValue( const int32_t i, const int32_t j, const int32_t k ) const
	{
		return m_Ptr[ ( j * m_Width + i ) * N + k ];
	}

	inline T getValueUV( const float u, const float v, const int32_t k ) const
	{
		const float _fi = u * m_Width;
		const float _fj = v * m_Height;

		const int i0 = floor(_fi);
		const int j0 = floor(_fj);

		const int i1 = i0 + 1;
		const int j1 = j0 + 1;

		if( ( i0 < 0 ) || ( i1 >= m_Width ) || ( j0 < 0 ) || ( j1 >= m_Height ) )
		{
			return 0.0;
		}

		const float s = _fi - i0;
		const float t = _fj - j0;

		const float v00 = m_Ptr[ ( j0 * m_Width + i0 ) * N + k ];
		const float v01 = m_Ptr[ ( j0 * m_Width + i1 ) * N + k ];
		const float v10 = m_Ptr[ ( j1 * m_Width + i0 ) * N + k ];
		const float v11 = m_Ptr[ ( j1 * m_Width + i1 ) * N + k ];

		const float val = t * ( s * v11 + ( 1.0 - s ) * v10 ) + ( 1.0 - t ) * ( s * v01 + ( 1.0 - s ) * v00 );
		return val;
	}

	inline T* getPtr()
	{
		return m_Ptr;
	}

	inline const T* getPtr() const
	{
		return m_Ptr;
	}

	inline int32_t getWidth() const
	{
		return m_Width;
	}

	inline int32_t getHeight() const
	{
		return m_Height;
	}
	
	void flipY()
	{
		for( int j=0; j<m_Height/2; j++ )
		{
			for( int i=0; i<m_Width; i++ )
			{
				for( int c=0; c<NC; c++ )
				{
					const int flat_idx_1 = j * m_Width * NC + i * NC + c;
					const int flat_idx_2 = ( m_Height - 1 - j ) * m_Width * NC + i * NC + c;
					T tmp = m_Ptr[flat_idx_1];
					m_Ptr[flat_idx_1] = m_Ptr[flat_idx_2];
					m_Ptr[flat_idx_2] = tmp;
				}
			}
		}
	}

protected:
	int32_t m_Width;
	int32_t m_Height;
	T* m_Ptr;

	enum { NC = N };
};

#endif
