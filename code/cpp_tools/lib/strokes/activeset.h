// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: activeset.h
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

#ifndef __activeset_h__
#define __activeset_h__

#include <lib/image/image.h>
#include <iostream>

class HierarchicalActiveSet
{
	HierarchicalActiveSet();
public:
	HierarchicalActiveSet( const int in_width, const int in_height );
	~HierarchicalActiveSet();
	
	void setActiveSetFromRGBABuffer( const Image<float, 4>& in_buffer, const float threshold );
	void setActiveSetFromRGBABufferBytes( const unsigned char* in_buffer, const int in_buffer_width, const int in_buffer_height, const float _threshold );	
	void updateActiveSetFromRGBABuffer( const Image<float, 4>& in_buffer, int in_x, int in_y, int in_w, int in_h, const float threshold );
	void updateActiveSetFromRGBABufferBytes( const unsigned char* in_buffer, const int in_buffer_width, int in_x, int in_y, int in_w, int in_h, const float _threshold );
	int findUnoccupied( const float in_random_number, int& out_x, int& out_y );
	void save_active_set_as_images( const std::string& in_filename_template );	
	void viewValues( const int in_x, const int in_y );	
	bool checkSummationConsistency();	
	void showNonZeros();
	
private:
	int m_Width;
	int m_Height;
	int m_BaseLevelWidth;
	int m_BaseLevelHeight;
	int m_TopLevelWidth;
	int m_TopLevelHeight;
	
	int m_numLevelsW;
	int m_numLevelsH;
	int m_numLevels;
	int** m_ActiveSet;
};

#endif