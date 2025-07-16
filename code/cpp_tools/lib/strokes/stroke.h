// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: stroke.h
// Maintainer: Yonghao Yue
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

#ifndef __stroke_h__
#define __stroke_h__

#include <Eigen/Core>
#include <vector>
#include "sortutil.h"
#include <iostream>

class Stroke
{
public:
	Stroke(){ std::cout << "WARNING!!! Calling the default constructor of Stroke class is unintended. This default constructor is made callable for the ease of implementation to shrink the size of StrokeData." << std::endl; exit(-1); };
	
	Stroke( const std::vector<Eigen::Vector2f>& in_center_line, const float in_width, const float* in_color4 )
		: m_Width( in_width ), m_num_tris( 0 )
	{
		for( int i=0; i<4; i++ )
			m_Color(i) = in_color4[i];
		
		m_CenterLine.resize( 2, in_center_line.size() );
		m_T.resize( 2, in_center_line.size() );
		m_B.resize( 2, in_center_line.size() );
		m_ArcParameter.resize( in_center_line.size() );
		
		for( int i=0; i<in_center_line.size(); i++ )
		{
			m_CenterLine.col(i) = in_center_line[i];
		}
		
		m_T.col(0) = ( in_center_line[1] - in_center_line[0] ).normalized();
		m_ArcParameter(0) = 0.0;
		for( int i=1; i<in_center_line.size()-1; i++ )
		{
			const Eigen::Vector2f E0 = ( in_center_line[i+1] - in_center_line[i] ).normalized();
			const Eigen::Vector2f _E1 = in_center_line[i] - in_center_line[i-1];
			const float E1_norm = _E1.norm();
			m_ArcParameter(i) = m_ArcParameter(i-1) + E1_norm;
			const Eigen::Vector2f E1 = _E1 / E1_norm;
			m_T.col(i) = ( 0.5 * ( E0 + E1 ) ).normalized();
		}
		
		const Eigen::Vector2f _E1 = in_center_line[ in_center_line.size()-1 ]  - in_center_line[ in_center_line.size()-2 ];
		const float E1_norm = _E1.norm();
		m_T.col(in_center_line.size()-1) = _E1 / E1_norm;
		m_ArcParameter(in_center_line.size()-1) = m_ArcParameter(in_center_line.size()-2) + E1_norm;
		
		for( int i=0; i<in_center_line.size(); i++ )
		{
			m_B.col(i).x() = m_T.col(i).y();
			m_B.col(i).y() = -m_T.col(i).x();
			m_ArcParameter(i) /= m_ArcParameter(in_center_line.size()-1);
		}
		
		m_num_tris = ( in_center_line.size() - 1 ) * 2;
		m_vertices.resize( m_num_tris * 3 * 2 );
		m_uvs.resize( m_num_tris * 3 * 2 );
		m_ts.resize( m_num_tris * 3 * 2 );
		
		// center line pos coord: [0, 1]^2
		// vertices coord: [-1, 1]^2
		
		Eigen::Vector2f P0 = m_CenterLine.col(0) - 0.5 * m_Width * m_B.col(0);
		Eigen::Vector2f P1 = m_CenterLine.col(0) + 0.5 * m_Width * m_B.col(0);
		
		for( int i=0; i<in_center_line.size()-1; i++ )
		{
			const Eigen::Vector2f P2 = m_CenterLine.col(i+1) - 0.5 * m_Width * m_B.col(i+1);
			const Eigen::Vector2f P3 = m_CenterLine.col(i+1) + 0.5 * m_Width * m_B.col(i+1);
			
			m_vertices( 2*i*3*2      ) = P0.x(); m_vertices( 2*i*3*2 + 1  ) = P0.y();
			m_vertices( 2*i*3*2 + 2  ) = P1.x(); m_vertices( 2*i*3*2 + 3  ) = P1.y();
			m_vertices( 2*i*3*2 + 4  ) = P3.x(); m_vertices( 2*i*3*2 + 5  ) = P3.y();
			m_vertices( 2*i*3*2 + 6  ) = P0.x(); m_vertices( 2*i*3*2 + 7  ) = P0.y();
			m_vertices( 2*i*3*2 + 8  ) = P3.x(); m_vertices( 2*i*3*2 + 9  ) = P3.y();
			m_vertices( 2*i*3*2 + 10 ) = P2.x(); m_vertices( 2*i*3*2 + 11 ) = P2.y();
			
			m_uvs( 2*i*3*2      ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 1  ) = 0.0;
			m_uvs( 2*i*3*2 + 2  ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 3  ) = 1.0;
			m_uvs( 2*i*3*2 + 4  ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 5  ) = 1.0;
			m_uvs( 2*i*3*2 + 6  ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 7  ) = 0.0;
			m_uvs( 2*i*3*2 + 8  ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 9  ) = 1.0;
			m_uvs( 2*i*3*2 + 10 ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 11 ) = 0.0;
			
			m_ts( 2*i*3*2      ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 1  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 2  ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 3  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 4  ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 5  ) = m_T.col(i+1).y();
			m_ts( 2*i*3*2 + 6  ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 7  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 8  ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 9  ) = m_T.col(i+1).y();
			m_ts( 2*i*3*2 + 10 ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 11 ) = m_T.col(i+1).y();
			
			P0 = P2;
			P1 = P3;
		}
		
		// BB coord: [0, 1]^2
		
		m_BBMin(0) = 1.0e33; m_BBMin(1) = 1.0e33;
		m_BBMax(0) = -1.0e33; m_BBMax(1) = -1.0e33;
		
		for( int i=0; i<m_num_tris*3; i++ )
		{
			m_BBMin(0) = std::min<float>( m_BBMin(0), m_vertices[ 2*i ] );
			m_BBMin(1) = std::min<float>( m_BBMin(1), m_vertices[ 2*i+1 ] );
			m_BBMax(0) = std::max<float>( m_BBMax(0), m_vertices[ 2*i ] );
			m_BBMax(1) = std::max<float>( m_BBMax(1), m_vertices[ 2*i+1 ] );
		}
		
		// [0, 1]^2 -> [-1, 1]2^
		for( int i=0; i<m_num_tris * 3 * 2; i++ )
			m_vertices(i) = 2.0 * m_vertices(i) - 1.0;
	}
	
	~Stroke()
	{}
	
	int numTris() const
	{
		return m_num_tris;
	}
	
	const float* vertices() const
	{
		return m_vertices.data();
	}
	
	const float* uvs() const
	{
		return m_uvs.data();
	}
	
	const float* ts() const
	{
		return m_ts.data();
	}
	
	void color( float out_color[4] ) const
	{
		for( int i=0; i<4; i++ )
			out_color[i] = m_Color(i);
	}
	
	const Eigen::Vector2f& BBMin() const
	{
		return m_BBMin;
	}
	
	const Eigen::Vector2f& BBMax() const
	{
		return m_BBMax;
	}
	
	const Eigen::Matrix2Xf& centerLine() const
	{
		return m_CenterLine;
	}
	
	float width() const
	{
		return m_Width;
	}
	
	const Eigen::Vector4f& color() const
	{
		return m_Color;
	}
		
private:
	Eigen::Matrix2Xf m_CenterLine;
	float m_Width;
	Eigen::Vector4f m_Color;
	
	Eigen::VectorXf m_vertices;
	Eigen::VectorXf m_uvs;
	Eigen::VectorXf m_ts;
	int m_num_tris;
	Eigen::Vector2f m_BBMin;
	Eigen::Vector2f m_BBMax;
	Eigen::Matrix2Xf m_T;
	Eigen::Matrix2Xf m_B;
	Eigen::VectorXf m_ArcParameter;
};

struct StrokeData
{
	std::vector<Stroke> strokes;
	std::vector<int> indices;
	std::vector<Eigen::Vector2i> tex_info_int;
	std::vector<Eigen::Vector2f> tex_info_float;
	
	void clear();
	void append( const StrokeData& in_tail );
	void rearrange( const std::vector<SortItem>& in_sort_result );
	void remove( const std::vector<int>& in_remove_flag );
	
	void shrinkSize( const float in_ratio );
};

struct TexAuxData
{
  Eigen::Vector2i size;
  int single_w;
  int single_h;
	float min_tex_ratio;
  int num_textures;
  int length_mipmap_level;
};

void showTexAuxData( const TexAuxData& in_data );

void compute_tex_settings( const Eigen::Vector2i& in_texture_size, const int in_num_textures, const int in_texture_length_mipmap_level,
	TexAuxData& out_tex_aux_data );
		
void saveStrokes( const std::string& in_h5_fn, const int in_width, const int in_height, const StrokeData& in_strokes, 
	const std::string& in_texture_filename, const TexAuxData& in_aux );
	
void loadStrokes( const std::string& in_h5_fn, int& out_width, int& out_height, StrokeData& out_strokes, 
	std::string& out_texture_filename, TexAuxData& out_aux );

float luminance( const Stroke& in_stroke );
float sindex( const int& in_stroke_index );

#endif