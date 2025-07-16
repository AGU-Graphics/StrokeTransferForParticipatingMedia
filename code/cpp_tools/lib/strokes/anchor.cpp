// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: anchor.cpp
// Maintainer: Yonghao Yue
//
// Description:
// This file implements data structure for anchor points, and provides procedures
// to append another set of anchor points, rearrange the anchor points according to
// a sorted order, remove anchor points according to flats, and save and load 
// anchor points.
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

#include "anchor.h"
#include <HDF5File.h>

void AnchorData::clear()
{
	pos.clear();
	born_time.clear();
	indices.clear();
	random_numbers.clear();
}

void AnchorData::append( const AnchorData& in_tail )
{
	pos.insert( pos.end(), in_tail.pos.begin(), in_tail.pos.end() );
	born_time.insert( born_time.end(), in_tail.born_time.begin(), in_tail.born_time.end() );
	indices.insert( indices.end(), in_tail.indices.begin(), in_tail.indices.end() );
	random_numbers.insert( random_numbers.end(), in_tail.random_numbers.begin(), in_tail.random_numbers.end() );
}
	
void AnchorData::rearrange( const std::vector<SortItem>& in_sort_result )
{
	std::vector<Eigen::Vector2f> _pos;
	std::vector<int> _born_time;
	std::vector<int> _indices;
	std::vector<Eigen::Matrix<float,6,1>> _random_numbers;
	
	rearrangeList( pos, _pos, in_sort_result );
	rearrangeList( born_time, _born_time, in_sort_result );
	rearrangeList( indices, _indices, in_sort_result );
	rearrangeList( random_numbers, _random_numbers, in_sort_result );
	pos.swap( _pos );
	born_time.swap( _born_time );
	indices.swap( _indices );
	random_numbers.swap( _random_numbers );
}

void AnchorData::remove( const std::vector<int>& in_remove_flag )
{
	const int orig_num_elems = pos.size();
	int c_new = 0;
	for( int i=0; i<orig_num_elems; i++ )
	{
		if( in_remove_flag[i] == 0 )
		{
			if( i != c_new )
			{
				pos[c_new] = pos[i];
				born_time[c_new] = born_time[i];
				indices[c_new] = indices[i];
				random_numbers[c_new] = random_numbers[i];
			}
			c_new++;
		}
	}

	pos.erase( pos.end() - ( orig_num_elems - c_new ), pos.end() );
	born_time.erase( born_time.end() - ( orig_num_elems - c_new ), born_time.end() );
	indices.erase( indices.end() - ( orig_num_elems - c_new ), indices.end() );
	random_numbers.erase( random_numbers.end() - ( orig_num_elems - c_new ), random_numbers.end() );
}

void AnchorData::appendFromOther( const AnchorData& in_other, const int in_anchor_idx )
{
	pos.push_back( in_other.pos[in_anchor_idx] );
	born_time.push_back( in_other.born_time[in_anchor_idx] );
	indices.push_back( in_other.indices[in_anchor_idx] );
	random_numbers.push_back( in_other.random_numbers[in_anchor_idx] );
}

void AnchorData::setFromOther( const AnchorData& in_other )
{
	clear();
	append( in_other );
}

void saveAnchors( const std::string& in_h5_fn, const int in_next_idx, const AnchorData& in_anchors )
{
	HDF5File h5( in_h5_fn, HDF5AccessType::READ_WRITE );
	h5.writeVector( "hpds", "num_levels", Eigen::Matrix<int,1,1>{ 1 } );
	h5.writeVector( "hpds", "r_lv0", Eigen::Matrix<float,1,1>{ -1.0 } );
	h5.writeVector( "hpds", "next_idx", Eigen::Matrix<int,1,1>{ in_next_idx } );
	
	std::string group_name = std::string( "hpds/lv" ) + std::to_string(0);
	h5.writeVector( group_name, "num_points", Eigen::Matrix<int,1,1>{ in_anchors.pos.size() } );
	Eigen::VectorXf x; x.resize( in_anchors.pos.size() );
	Eigen::VectorXf y; y.resize( in_anchors.pos.size() );
	Eigen::VectorXi born_time; born_time.resize( in_anchors.pos.size() );
	Eigen::VectorXi idx; born_time.resize( in_anchors.indices.size() );
	Eigen::MatrixXf random_numbers; random_numbers.resize( in_anchors.indices.size(), 6 );
	for( int i=0; i<in_anchors.pos.size(); i++ )
	{
		x(i) = in_anchors.pos[i].x();
		y(i) = in_anchors.pos[i].y();
		born_time(i) = in_anchors.born_time[i];
		idx(i) = in_anchors.indices[i];
		random_numbers.row(i) = in_anchors.random_numbers[i];
	}
	h5.writeVector( group_name, "x", x );
	h5.writeVector( group_name, "y", y );
	h5.writeVector( group_name, "born_frame", born_time );
	h5.writeVector( group_name, "idx", born_time );
	h5.writeMatrix( group_name, "random_numbers", random_numbers );
}

void loadAnchors( const std::string& in_h5_fn, int& out_next_idx, AnchorData& out_anchors )
{
	out_anchors.clear();
	
	HDF5File h5( in_h5_fn, HDF5AccessType::READ_ONLY );
	Eigen::Matrix<int,1,1> num_levels;
	h5.readVector( "hpds", "num_levels", num_levels );
	Eigen::Matrix<float,1,1> r_lv0;
	h5.readVector( "hpds", "r_lv0", r_lv0 );
	Eigen::Matrix<int,1,1> next_idx;
	h5.readVector( "hpds", "next_idx", next_idx );
	out_next_idx = next_idx( 0, 0 );
	
	std::string group_name = std::string( "hpds/lv" ) + std::to_string(0);
	Eigen::Matrix<int,1,1> num_points;
	h5.readVector( group_name, "num_points", num_points );
		
	Eigen::VectorXf x;
	Eigen::VectorXf y;
	Eigen::VectorXi born_time;
	Eigen::VectorXi idx;
	Eigen::MatrixXf random_numbers;
	
	h5.readVector( group_name, "x", x );
	h5.readVector( group_name, "y", y );
	h5.readVector( group_name, "born_frame", born_time );
	h5.readVector( group_name, "idx", idx );
	h5.readMatrix( group_name, "random_numbers", random_numbers );
	
	for( int i=0; i<num_points( 0, 0 ); i++ )
	{
		out_anchors.pos.emplace_back( x(i), y(i) );
		out_anchors.born_time.push_back( born_time(i) );
		out_anchors.indices.push_back( idx(i) );
		out_anchors.random_numbers.push_back( random_numbers.row(i) );
	}
}
