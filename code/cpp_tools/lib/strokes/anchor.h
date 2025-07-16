// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: anchor.h
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

#ifndef __anchor_h__
#define __anchor_h__

#include <iostream>
#include <Eigen/Core>
#include <vector>
#include "sortutil.h"

struct AnchorData
{
	std::vector<Eigen::Vector2f> pos;
	std::vector<int> born_time;
	std::vector<int> indices;
	std::vector<Eigen::Matrix<float,6,1>> random_numbers;
	
	void clear();
	void append( const AnchorData& in_tail );
	void rearrange( const std::vector<SortItem>& in_sort_result );
	void remove( const std::vector<int>& in_remove_flag );
	
	void appendFromOther( const AnchorData& in_other, const int in_anchor_idx );
	void setFromOther( const AnchorData& in_other );
};

void saveAnchors( const std::string& in_h5_fn, const int in_next_idx, const AnchorData& in_anchors );
void loadAnchors( const std::string& in_h5_fn, int& out_next_idx, AnchorData& out_anchors );

#endif