// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: sortutil.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides helper functions for performing sorting.
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

#ifndef __sortutil_h__
#define __sortutil_h__

#include <vector>

struct SortItem
{
	SortItem( float _value, int _index )
		: value( _value ), index( _index )
	{}
	
	float value;
	int index;
};

inline bool operator<( const SortItem& in_lhs, const SortItem& in_rhs )
{
	return in_lhs.value < in_rhs.value || ( in_lhs.value == in_rhs.value && in_lhs.index < in_rhs.index );
}

template<typename T>
void buildSortItemList( const std::vector<T>& in_data, const std::function<float(const T&)>& in_func, std::vector<SortItem>& out_list )
{
	out_list.clear();
	for( int i=0; i<in_data.size(); i++ )
		out_list.emplace_back( in_func( in_data[i] ), i );
}

template<typename T>
void rearrangeList( const std::vector<T>& in_data, std::vector<T>& out_data, const std::vector<SortItem>& in_sort_result )
{
	if( in_data.size() == 0 ) return;
	out_data.resize( in_data.size(), in_data[0] );
	for( int i=0; i<in_data.size(); i++ )
		out_data[i] = in_data[ in_sort_result[i].index ];
}

#endif