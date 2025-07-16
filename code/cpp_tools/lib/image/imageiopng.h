// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: imageiopng.h
// Maintainer: Yonghao Yue
//
// Description:
// This file implements io for images via the png format.
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

#ifndef image_io_png_h
#define image_io_png_h

#include "./image.h"
#include "./imageioutil.h"

#include <png.h>

#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

struct PngIO
{
	PngIO()
	: png_ptr( nullptr ), info_ptr( nullptr ), image_data( nullptr ), width( 0 ), height( 0 ), bit_depth( 0 ), color_type( 0 ), interlace_type( PNG_INTERLACE_NONE )
	{}

	png_structp png_ptr;
	png_infop info_ptr;
	int bit_depth;
	int color_type;
	int interlace_type;
	png_uint_32 width;
	png_uint_32 height;
	unsigned char* image_data;
};

inline int readpng_init( FILE* f, long * pWidth, long* pHeight, PngIO& png_io )
{
	unsigned char sig[8];

	fread( sig, 1, 8, f );
	if( !png_check_sig( sig, 8 ) )
		return 1;

	png_io.png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
	if( !png_io.png_ptr )
		return 4;   /* out of memory */

	png_io.info_ptr = png_create_info_struct( png_io.png_ptr );
	if( !png_io.info_ptr )
	{
		png_destroy_read_struct( &png_io.png_ptr, NULL, NULL );
		return 4;   /* out of memory */
	}

	if( setjmp( png_jmpbuf( png_io.png_ptr ) ) )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		return 2;
	}

	png_init_io( png_io.png_ptr, f );
	png_set_sig_bytes( png_io.png_ptr, 8 );
	png_read_info( png_io.png_ptr, png_io.info_ptr );

	png_get_IHDR( png_io.png_ptr, png_io.info_ptr, &png_io.width, &png_io.height, &png_io.bit_depth, &png_io.color_type, &png_io.interlace_type, NULL, NULL );
	*pWidth = png_io.width;
	*pHeight = png_io.height;

	return 0;
}

inline int readpng_get_bgcolor( unsigned char *red, unsigned char *green, unsigned char *blue, PngIO& png_io )
{
	png_color_16p pBackground;

	if( setjmp( png_jmpbuf( png_io.png_ptr ) ) )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		return 2;
	}

	if( !png_get_valid( png_io.png_ptr, png_io.info_ptr, PNG_INFO_bKGD ) )
		return 1;

	png_get_bKGD( png_io.png_ptr, png_io.info_ptr, &pBackground );

	if( png_io.bit_depth == 16 )
	{
		*red   = pBackground->red   >> 8;
		*green = pBackground->green >> 8;
		*blue  = pBackground->blue  >> 8;
	}
	else if( png_io.color_type == PNG_COLOR_TYPE_GRAY && png_io.bit_depth < 8 )
	{
		if( png_io.bit_depth == 1 )
			*red = *green = *blue = pBackground->gray? 255 : 0;
		else if( png_io.bit_depth == 2 )
			*red = *green = *blue = ( 255 / 3 ) * pBackground->gray;
		else // bit_depth == 4
			*red = *green = *blue = ( 255 / 15 ) * pBackground->gray;
	}
	else
	{
		*red   = ( unsigned char ) pBackground->red;
		*green = ( unsigned char ) pBackground->green;
		*blue  = ( unsigned char ) pBackground->blue;
	}

	return 0;
}

inline unsigned char* readpng_get_image( double display_exponent, int *pChannels, unsigned long *pRowbytes, PngIO& png_io )
{
	double gamma;
	png_uint_32  i, rowbytes;
	png_bytepp  row_pointers = NULL;

	if( setjmp( png_jmpbuf( png_io.png_ptr ) ) )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		return NULL;
	}

	if( png_io.color_type == PNG_COLOR_TYPE_PALETTE )
		png_set_expand( png_io.png_ptr );
	if( png_io.color_type == PNG_COLOR_TYPE_GRAY && png_io.bit_depth < 8 )
		png_set_expand( png_io.png_ptr );
	if( png_get_valid( png_io.png_ptr, png_io.info_ptr, PNG_INFO_tRNS ) )
		png_set_expand( png_io.png_ptr );
	if( png_io.bit_depth == 16 )
		png_set_strip_16( png_io.png_ptr );
	if( png_io.color_type == PNG_COLOR_TYPE_GRAY || png_io.color_type == PNG_COLOR_TYPE_GRAY_ALPHA )
		png_set_gray_to_rgb( png_io.png_ptr );

	if( png_get_gAMA( png_io.png_ptr, png_io.info_ptr, &gamma ) )
		png_set_gamma( png_io.png_ptr, display_exponent, gamma );

	png_read_update_info( png_io.png_ptr, png_io.info_ptr );

	*pRowbytes = rowbytes = png_get_rowbytes( png_io.png_ptr, png_io.info_ptr );
	*pChannels = (int) png_get_channels( png_io.png_ptr, png_io.info_ptr );

	if( ( png_io.image_data = ( unsigned char* ) malloc( rowbytes * png_io.height ) ) == NULL )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		return NULL;
	}
	if( ( row_pointers = ( png_bytepp ) malloc( png_io.height * sizeof( png_bytep ) ) ) == NULL )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		free( png_io.image_data );
		png_io.image_data = NULL;
		return NULL;
	}

	for( i = 0; i < png_io.height; ++i )
		row_pointers[i] = png_io.image_data + i * rowbytes;

	png_read_image( png_io.png_ptr, row_pointers );

	free( row_pointers );
	row_pointers = NULL;

	png_read_end( png_io.png_ptr, NULL );

	return png_io.image_data;
}

inline void readpng_cleanup( int free_image_data, PngIO& png_io )
{
	if( free_image_data && png_io.image_data )
	{
		free( png_io.image_data );
		png_io.image_data = NULL;
	}

	if( png_io.png_ptr && png_io.info_ptr )
	{
		png_destroy_read_struct( &png_io.png_ptr, &png_io.info_ptr, NULL );
		png_io.png_ptr = NULL;
		png_io.info_ptr = NULL;
	}
}

template<typename T, int N>
struct cRGB2color
{
	bool operator()( unsigned char rgb[3], T c[N] ) { return false; }
};

template<typename T>
struct cRGB2color<T, 1>
{
	bool operator()( unsigned char rgb[3], T c[1] )
	{
		c[0] = ( double( rgb[0] ) + double( rgb[1] ) + double( rgb[2] ) ) / ( 3.0 * 255.0 );
		return true;
	}
};

template<typename T>
struct cRGB2color<T, 2>
{
	bool operator()( unsigned char rgb[3], T c[2] )
	{
		c[0] = double( rgb[0] ) / 255.0;
		c[1] = double( rgb[1] ) / 255.0;
		return true;
	}
};

template<typename T>
struct cRGB2color<T, 3>
{
	bool operator()( unsigned char rgb[3], T c[3] )
	{
		c[0] = double( rgb[0] ) / 255.0;
		c[1] = double( rgb[1] ) / 255.0;
		c[2] = double( rgb[2] ) / 255.0;
		return true;
	}
};

template<typename T>
struct cRGB2color<T, 4>
{
	bool operator()( unsigned char rgb[3], T c[4] )
	{
		c[0] = double( rgb[0] ) / 255.0;
		c[1] = double( rgb[1] ) / 255.0;
		c[2] = double( rgb[2] ) / 255.0;
		c[3] = 1.0;
		return true;
	}
};

template<typename T, int N>
struct cRGBA2color
{
	bool operator()( unsigned char rgba[4], T c[N] ) { return false; }
};

template<typename T>
struct cRGBA2color<T, 1>
{
	bool operator()( unsigned char rgba[4], T c[1] )
	{
		c[0] = ( double( rgba[0] ) + double( rgba[1] ) + double( rgba[2] ) ) / ( 3.0 * 255.0 );
		return true;
	}
};

template<typename T>
struct cRGBA2color<T, 2>
{
	bool operator()( unsigned char rgba[4], T c[2] )
	{
		c[0] = double( rgba[0] ) / 255.0;
		c[1] = double( rgba[1] ) / 255.0;
		return true;
	}
};

template<typename T>
struct cRGBA2color<T, 3>
{
	bool operator()( unsigned char rgba[4], T c[3] )
	{
		c[0] = double( rgba[0] ) / 255.0;
		c[1] = double( rgba[1] ) / 255.0;
		c[2] = double( rgba[2] ) / 255.0;
		return true;
	}
};

template<typename T>
struct cRGBA2color<T, 4>
{
	bool operator()( unsigned char rgba[4], T c[4] )
	{
		c[0] = double( rgba[0] ) / 255.0;
		c[1] = double( rgba[1] ) / 255.0;
		c[2] = double( rgba[2] ) / 255.0;
		c[3] = double( rgba[3] ) / 255.0;
		return true;
	}
};

template<typename T, int N>
inline bool loadImagePng( const std::string& in_Filename, Image<T, N>& io_Image )
{
	FILE* f = fopen( in_Filename.c_str(), "rb" );
	if( !f )
	{
		std::cerr << "can't open PNG file " << in_Filename << std::endl;
		return false;
	}

	PngIO png_io;

	long width, height;
	int rc = readpng_init( f, &width, &height, png_io );
	if( rc != 0 )
	{
		switch( rc )
		{
		case 1:
      std::cerr << in_Filename << " is not a PNG file: incorrect signature" << std::endl;
      break;
    case 2:
			std::cerr << in_Filename << " has bad IHDR (libpng longjmp)" << std::endl;
			break;
    case 4:
      std::cerr << "insufficient memory" << std::endl;
      break;
    default:
      std::cerr << "unknown readpng_init() error" << std::endl;
      break;
		}

		fclose( f );

		return false;
	}
	
	if( png_io.interlace_type != PNG_INTERLACE_NONE )
		png_set_interlace_handling( png_io.png_ptr );

	unsigned long image_rowbytes;
	int image_channels;

	unsigned char* image_data = readpng_get_image( 2.2, &image_channels, &image_rowbytes, png_io );

  io_Image.init( width, height );

	unsigned char* p_img_data = image_data;

	if( image_channels == 3 )
	{
		for( int i=0; i<width*height; i++ )
		{
			unsigned char data[3] = { *p_img_data++, *p_img_data++, *p_img_data++ };
			cRGB2color<T, N>()( data, &( io_Image.getPtr()[i*N] ) );
		}
	}
	else
	{
		for( int i=0; i<width*height; i++ )
		{
			unsigned char data[4] = { *p_img_data++, *p_img_data++, *p_img_data++, *p_img_data++ };
			cRGBA2color<T, N>()( data, &( io_Image.getPtr()[i*N] ) );
		}
	}

	readpng_cleanup( 1, png_io );

	return true;
}

template<typename T, int N>
inline bool saveImagePng( const std::string& in_Filename, const Image<T, N>& in_Image )
{
	uint8_t* lineBuffers = (uint8_t*)malloc(sizeof(uint8_t) * 4 * in_Image.getWidth() * in_Image.getHeight());
	png_structp png = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
	png_infop info = png_create_info_struct( png );
	png_bytep* lines = NULL;
	FILE* f = fopen( in_Filename.c_str(), "wb" );
	png_init_io( png, f );
	png_set_IHDR( png, info, in_Image.getWidth(), in_Image.getHeight(), 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_BASE );

	png_color_8 sBIT;
	sBIT.red = 8;
	sBIT.green = 8;
	sBIT.blue = 8;
	sBIT.alpha = 8;
	png_set_sBIT( png, info, &sBIT );

	png_write_info( png, info );
	//png_set_bgr(png);

	lines = ( png_bytep* ) malloc( sizeof( png_bytep ) * in_Image.getHeight() );

	for( int j=0; j<in_Image.getHeight(); j++ )
	{
		for( int i=0; i<in_Image.getWidth(); i++ )
		{
			color2RGBA<T, N>()( &in_Image.getPtr()[ ( j * in_Image.getWidth() + i ) * N ], &lineBuffers[ ( j * in_Image.getWidth() + i ) * 4 ] );
		}
		lines[j] = ( png_bytep ) &lineBuffers[ ( j * in_Image.getWidth() ) * 4 ];
	}

	png_write_image( png, lines );
	png_write_end( png, info );
	png_destroy_write_struct( &png, &info );
	free( lineBuffers );
	free( lines );
	fclose( f );

	return true;
}

template<typename T, int N>
inline bool saveImagePngInvertY( const std::string& in_Filename, const Image<T, N>& in_Image )
{
	uint8_t* lineBuffers = (uint8_t*)malloc(sizeof(uint8_t) * 4 * in_Image.getWidth() * in_Image.getHeight());
	png_structp png = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
	png_infop info = png_create_info_struct( png );
	png_bytep* lines = NULL;
	FILE* f = fopen( in_Filename.c_str(), "wb" );
	png_init_io( png, f );
	png_set_IHDR( png, info, in_Image.getWidth(), in_Image.getHeight(), 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_BASE );

	png_color_8 sBIT;
	sBIT.red = 8;
	sBIT.green = 8;
	sBIT.blue = 8;
	sBIT.alpha = 8;
	png_set_sBIT( png, info, &sBIT );

	png_write_info( png, info );

	lines = ( png_bytep* ) malloc( sizeof( png_bytep ) * in_Image.getHeight() );

	for( int j=0; j<in_Image.getHeight(); j++ )
	{
		for( int i=0; i<in_Image.getWidth(); i++ )
		{
			color2RGBA<T, N>()( &in_Image.getPtr()[ ( ( in_Image.getHeight() - 1 - j ) * in_Image.getWidth() + i ) * N ], &lineBuffers[ ( j * in_Image.getWidth() + i ) * 4 ] );
		}
		lines[j] = ( png_bytep ) &lineBuffers[ ( j * in_Image.getWidth() ) * 4 ];
	}

	png_write_image( png, lines );
	png_write_end( png, info );
	png_destroy_write_struct( &png, &info );
	free( lineBuffers );
	free( lines );
	fclose( f );

	return true;
}

#endif
