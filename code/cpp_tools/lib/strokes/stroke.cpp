// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: stroke.cpp
// Maintainer: Yonghao Yue and Hideki Todo
//
// Description:
// This file defines the Stroke class, which generates stroke polygons based on
// input centerline, width, and color information. It also provides utilities for
// managing stroke lists, including appending another list, reordering according
// to a given index sequence, removing strokes using flat indices, and saving/loading
// stroke data.
//
// For stroke textures, the default mode uses a single shared texture across all strokes.
// Alternatively, a "tiled" mode is supported, enabling style variation and mipmapping:
// - Vertically tiled textures introduce stylistic variations: the rendering algorithm 
//   will randomly assign different vertical slices to individual strokes.
// - Horizontally tiled textures implement a mipmap-like structure along the stroke's
//   length, where each consecutive tile (left to right) halves the texture width
//   relative to its predecessor.
//
// Tiling configuration is specified via the TexAuxData structure.
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

#include "stroke.h"
#include <HDF5File.h>

void StrokeData::clear()
{
	strokes.clear();
	indices.clear();
	tex_info_int.clear();
	tex_info_float.clear();
}

void StrokeData::append( const StrokeData& in_tail )
{
	strokes.insert( strokes.end(), in_tail.strokes.begin(), in_tail.strokes.end() );
	indices.insert( indices.end(), in_tail.indices.begin(), in_tail.indices.end() );
	tex_info_int.insert( tex_info_int.end(), in_tail.tex_info_int.begin(), in_tail.tex_info_int.end() );
	tex_info_float.insert( tex_info_float.end(), in_tail.tex_info_float.begin(), in_tail.tex_info_float.end() );
}

void StrokeData::rearrange( const std::vector<SortItem>& in_sort_result )
{
	std::vector<Stroke> _strokes;
	std::vector<int> _indices;
	std::vector<Eigen::Vector2i> _tex_info_int;
	std::vector<Eigen::Vector2f> _tex_info_float;
	
	rearrangeList( strokes, _strokes, in_sort_result );
	rearrangeList( indices, _indices, in_sort_result );
	rearrangeList( tex_info_int, _tex_info_int, in_sort_result );
	rearrangeList( tex_info_float, _tex_info_float, in_sort_result );
	strokes.swap( _strokes );
	indices.swap( _indices );
	tex_info_int.swap( _tex_info_int );
	tex_info_float.swap( _tex_info_float );
}

void StrokeData::remove( const std::vector<int>& in_remove_flag )
{
	const int orig_num_elems = strokes.size();
	int c_new = 0;
	for( int i=0; i<orig_num_elems; i++ )
	{
		if( in_remove_flag[i] == 0 )
		{
			if( i != c_new )
			{
				strokes[c_new] = strokes[i];
				indices[c_new] = indices[i];
				tex_info_int[c_new] = tex_info_int[i];
				tex_info_float[c_new] = tex_info_float[i];
			}
			c_new++;
		}
	}

	strokes.erase( strokes.end() - ( orig_num_elems - c_new ), strokes.end() );
	indices.erase( indices.end() - ( orig_num_elems - c_new ), indices.end() );
	tex_info_int.erase( tex_info_int.end() - ( orig_num_elems - c_new ), tex_info_int.end() );
	tex_info_float.erase( tex_info_float.end() - ( orig_num_elems - c_new ), tex_info_float.end() );
}

void StrokeData::shrinkSize( const float in_ratio )
{
	if( in_ratio >= 1.0 || in_ratio < 0 )
		return;
	
	const int newSize = int( strokes.size() * in_ratio );
	strokes.resize( newSize );
	indices.resize( newSize );
	tex_info_int.resize( newSize );
	tex_info_float.resize( newSize );
}

void showTexAuxData( const TexAuxData& in_data )
{
	std::cout << "texauxdata: " << std::endl;
	std::cout << in_data.size.transpose() << std::endl;
	std::cout << in_data.single_w << ", " << in_data.single_h << ", " << in_data.min_tex_ratio << std::endl;
	std::cout << in_data.num_textures << ", " << in_data.length_mipmap_level << std::endl;
}

void compute_tex_settings( const Eigen::Vector2i& in_texture_size, const int in_num_textures, const int in_texture_length_mipmap_level,
	TexAuxData& out_tex_aux_data )
{
  int num_tex_units = 0;
  int unit_size = 1;
  for( int i=0; i<in_texture_length_mipmap_level; i++ )
	{
    num_tex_units += unit_size;
    unit_size *= 2;
	}
	
	out_tex_aux_data.size = in_texture_size;
	out_tex_aux_data.single_w = int( in_texture_size(0) / num_tex_units );
  out_tex_aux_data.single_h = int( in_texture_size(1) / in_num_textures );
	out_tex_aux_data.min_tex_ratio = out_tex_aux_data.single_w / out_tex_aux_data.single_h;
  out_tex_aux_data.num_textures = in_num_textures;
  out_tex_aux_data.length_mipmap_level = in_texture_length_mipmap_level;
    
	if( out_tex_aux_data.single_w * num_tex_units != in_texture_size(0) || out_tex_aux_data.single_h * in_num_textures != in_texture_size(1) )
    std::cout << "Illegal texture size. Please double check the texture content, as well as the settings for num_textures and texture_length_mipmap_level." << std::endl;      
}

void saveStrokes( const std::string& in_h5_fn, const int in_width, const int in_height, const StrokeData& in_strokes, 
	const std::string& in_texture_filename, const TexAuxData& in_aux )
{
	HDF5File h5( in_h5_fn, HDF5AccessType::READ_WRITE );
	Eigen::Vector2i resolution = Eigen::Vector2i{ in_width, in_height };
	h5.writeVector( "strokes", "resolution", resolution );
	h5.writeString( "strokes", "texture_filename", in_texture_filename );
	h5.writeVector( "strokes/tex_aux", "shape", Eigen::Vector3i{ in_aux.size(1), in_aux.size(0), 4 } );
	h5.writeVector( "strokes/tex_aux", "single_size", Eigen::Vector2i{ in_aux.single_w, in_aux.single_h } );
	h5.writeVector( "strokes/tex_aux", "num_textures", Eigen::Matrix<int,1,1>{ in_aux.num_textures } );
	h5.writeVector( "strokes/tex_aux", "length_mipmap_level", Eigen::Matrix<int,1,1>{ in_aux.length_mipmap_level } );
	h5.writeVector( "strokes/tex_aux", "min_tex_ratio", Eigen::Matrix<float,1,1>{ in_aux.min_tex_ratio } );
	Eigen::MatrixXf stroke_tex_info; stroke_tex_info.resize( in_strokes.tex_info_int.size(), 4 );
	for( int i=0; i<in_strokes.tex_info_int.size(); i++ )
	{
		stroke_tex_info( i, 0 ) = in_strokes.tex_info_int[i](0);
		stroke_tex_info( i, 1 ) = in_strokes.tex_info_int[i](1);
		stroke_tex_info( i, 2 ) = in_strokes.tex_info_float[i](0);
		stroke_tex_info( i, 3 ) = in_strokes.tex_info_float[i](1);
	}
	h5.writeMatrix( "strokes", "stroke_tex_info", stroke_tex_info );
	Eigen::VectorXi stroke_indices; stroke_indices.resize( in_strokes.indices.size() );
	for( int i=0; i<in_strokes.indices.size(); i++ )
		stroke_indices(i) = in_strokes.indices[i];
	h5.writeVector( "strokes", "stroke_indices", stroke_indices );
	h5.writeVector( "strokes", "num_strokes", Eigen::Matrix<int,1,1>{ in_strokes.strokes.size() } );
	for( int i=0; i<in_strokes.strokes.size(); i++ )
	{
		std::string group_name = std::string( "strokes/stroke" ) + std::to_string(i);
		h5.writeMatrix( group_name.c_str(), "center_line", in_strokes.strokes[i].centerLine() );
		h5.writeVector( group_name.c_str(), "width", Eigen::Matrix<float,1,1>{ in_strokes.strokes[i].width() } );
		h5.writeVector( group_name.c_str(), "color", in_strokes.strokes[i].color() );
	}
}

void loadStrokes( const std::string& in_h5_fn, int& out_width, int& out_height, StrokeData& out_strokes, 
	std::string& out_texture_filename, TexAuxData& out_aux )
{
	HDF5File h5( in_h5_fn, HDF5AccessType::READ_ONLY );
	Eigen::Vector2i resolution;
	h5.readVector( "strokes", "resolution", resolution );
	out_width = resolution(0);
	out_height = resolution(1);
	h5.readString( "strokes", "texture_filename", out_texture_filename );
	
	Eigen::Vector3i tex_aux_shape;
	h5.readVector( "strokes/tex_aux", "shape", tex_aux_shape );
	out_aux.size(0) = tex_aux_shape(1); out_aux.size(1) = tex_aux_shape(0);
	
	Eigen::Vector3i tex_aux_single_size;
	h5.readVector( "strokes/tex_aux", "single_size", tex_aux_single_size );
	out_aux.single_w = tex_aux_single_size(0); out_aux.single_h = tex_aux_single_size(1); 
	
	Eigen::Matrix<int,1,1> tex_aux_num_textures;
	h5.readVector( "strokes/tex_aux", "num_textures", tex_aux_num_textures );
	out_aux.num_textures = tex_aux_num_textures(0);
	
	Eigen::Matrix<int,1,1> tex_aux_length_mipmap_level;
	h5.readVector( "strokes/tex_aux", "length_mipmap_level", tex_aux_length_mipmap_level );
	out_aux.length_mipmap_level = tex_aux_length_mipmap_level(0);
	
	Eigen::Matrix<float,1,1> tex_aux_min_tex_ratio;
	h5.readVector( "strokes/tex_aux", "min_tex_ratio", tex_aux_min_tex_ratio );
	out_aux.min_tex_ratio = tex_aux_min_tex_ratio(0);
	
	Eigen::MatrixXf stroke_tex_info;
	h5.readMatrix( "strokes", "stroke_tex_info", stroke_tex_info );
	
	out_strokes.tex_info_int.resize( stroke_tex_info.rows() );
	out_strokes.tex_info_float.resize( stroke_tex_info.rows() );
	
	for( int i=0; i<stroke_tex_info.rows(); i++ )
	{
		out_strokes.tex_info_int[i][0] = int( floor( 0.5 + stroke_tex_info( i, 0 ) ) );
		out_strokes.tex_info_int[i][1] = int( floor( 0.5 + stroke_tex_info( i, 1 ) ) );
		out_strokes.tex_info_float[i][0] = stroke_tex_info( i, 2 );
		out_strokes.tex_info_float[i][1] = stroke_tex_info( i, 3 );
	}
	
	Eigen::VectorXi stroke_indices;
	h5.readVector( "strokes", "stroke_indices", stroke_indices );
	out_strokes.indices.resize( stroke_indices.size() );
	for( int i=0; i<stroke_indices.size(); i++ )
		out_strokes.indices[i] = stroke_indices(i);
	
	Eigen::Matrix<int,1,1> num_strokes;
	h5.readVector( "strokes", "num_strokes", num_strokes );
	
	out_strokes.strokes.clear();
	for( int i=0; i<num_strokes(0); i++ )
	{
		std::string group_name = std::string( "strokes/stroke" ) + std::to_string(i);
		Eigen::Matrix2Xf _centerLine;
		h5.readMatrix( group_name.c_str(), "center_line", _centerLine );
		Eigen::Matrix<float,1,1> _width;
		h5.readVector( group_name.c_str(), "width", _width );
		Eigen::Vector4f _color;
		h5.readVector( group_name.c_str(), "color", _color );
		
		std::vector<Eigen::Vector2f> center_line_vect; center_line_vect.resize( _centerLine.cols() );
		for( int j=0; j<_centerLine.cols(); j++ )
			center_line_vect[j] = _centerLine.col(j);
		
		out_strokes.strokes.emplace_back( center_line_vect, _width(0), _color.data() );
	}
}

float luminance( const Stroke& in_stroke )
{
	// https://www.quora.com/What-is-the-equation-to-convert-RGB-to-Lab
	const float r = in_stroke.color()(0);
	const float g = in_stroke.color()(1);
	const float b = in_stroke.color()(2);
	
	const float r_linear = r <= 0.04045 ? r / 12.92 : std::pow( ( r + 0.055 ) / 1.055, 2.4 ); 
	const float g_linear = g <= 0.04045 ? g / 12.92 : std::pow( ( g + 0.055 ) / 1.055, 2.4 ); 
	const float b_linear = b <= 0.04045 ? b / 12.92 : std::pow( ( b + 0.055 ) / 1.055, 2.4 );
	
	const float X = 0.4124564 * r_linear + 0.3575761 * g_linear + 0.1804375 * b_linear;
	const float Y = 0.2126729 * r_linear + 0.7151522 * g_linear + 0.0721750 * b_linear;
	const float Z = 0.0193339 * r_linear + 0.1191920 * g_linear + 0.9503041 * b_linear;
	
	const float X65 = 0.95047; const float Y65 = 1.0; const float Z65 = 1.08883;
	const float Xn = X / X65; const float Yn = Y / Y65; const float Zn = Z / Z65;
	return 1.16 * std::pow( Yn, 1.0/3.0 ) - 0.16;	
}

float sindex( const int& in_stroke_index )
{
	return in_stroke_index;	
}