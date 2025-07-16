// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: imageioutil.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides helper functions for io for images.
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

#ifndef image_io_util_h
#define image_io_util_h

#include <stdint.h>
#include <cmath>
#include <iostream>

template<typename T, int N>
struct color2RGB
{
	bool operator()( const T c[N], uint8_t rgb[3] ) { return false; }
};

template<typename T>
struct color2RGB<T, 1>
{
	bool operator()( const T c[1], uint8_t rgb[3] )
	{
		int v = std::max<int>( 0, std::min<int>( 255, int( c[0] * 256 ) ) );
		rgb[0] = rgb[1] = rgb[2] = v;
		return true;
	}
};

template<typename T>
struct color2RGB<T, 2>
{
	bool operator()( const T c[2], uint8_t rgb[3] )
	{
		int v1 = std::max( 0, std::min( 255, int( c[0] * 256 ) ) );
		int v2 = std::max( 0, std::min( 255, int( c[1] * 256 ) ) );
		rgb[0] = v1;
		rgb[1] = v2;
		rgb[2] = 0;
		return true;
	}
};

template<typename T>
struct color2RGB<T, 3>
{
	bool operator()( const T c[3], uint8_t rgb[3] )
	{
		int v1 = std::max( 0, std::min( 255, int( c[0] * 256 ) ) );
		int v2 = std::max( 0, std::min( 255, int( c[1] * 256 ) ) );
		int v3 = std::max( 0, std::min( 255, int( c[2] * 256 ) ) );
		rgb[0] = v1;
		rgb[1] = v2;
		rgb[2] = v3;
		return true;
	}
};

template<typename T, int N>
struct color2RGBA
{
	bool operator()( const T c[N], uint8_t rgba[4] ) { return false; }
};

template<typename T>
struct color2RGBA<T, 1>
{
	bool operator()( const T c[1], uint8_t rgba[4] )
	{
		int v = std::max<int>( 0, std::min<int>( 255, int( c[0] * 256 ) ) );
		rgba[0] = rgba[1] = rgba[2] = v;
		rgba[3] = 255;
		return true;
	}
};

template<typename T>
struct color2RGBA<T, 2>
{
	bool operator()( const T c[2], uint8_t rgba[4] )
	{
		int v1 = std::max( 0, std::min( 255, int( c[0] * 256 ) ) );
		int v2 = std::max( 0, std::min( 255, int( c[1] * 256 ) ) );
		rgba[0] = v1;
		rgba[1] = v2;
		rgba[2] = 0;
		rgba[3] = 255;
		return true;
	}
};

template<typename T>
struct color2RGBA<T, 3>
{
	bool operator()( const T c[3], uint8_t rgba[4] )
	{
		int v1 = std::max( 0, std::min( 255, int( c[0] * 256 ) ) );
		int v2 = std::max( 0, std::min( 255, int( c[1] * 256 ) ) );
		int v3 = std::max( 0, std::min( 255, int( c[2] * 256 ) ) );
		rgba[0] = v1;
		rgba[1] = v2;
		rgba[2] = v3;
		rgba[3] = 255;
		return true;
	}
};

template<typename T>
struct color2RGBA<T, 4>
{
	bool operator()( const T c[4], uint8_t rgba[4] )
	{
		int v1 = std::max( 0, std::min( 255, int( c[0] * 256 ) ) );
		int v2 = std::max( 0, std::min( 255, int( c[1] * 256 ) ) );
		int v3 = std::max( 0, std::min( 255, int( c[2] * 256 ) ) );
		int v4 = std::max( 0, std::min( 255, int( c[3] * 256 ) ) );
		rgba[0] = v1;
		rgba[1] = v2;
		rgba[2] = v3;
		rgba[3] = v4;
		return true;
	}
};

#endif
