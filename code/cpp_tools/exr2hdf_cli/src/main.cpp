// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: exr2hdf_cli/src/main.cpp
// Maintainer: Yonghao Yue
//
// Description:
// This code converts an exr file to our internal hdf5 format for later ease 
// of data processing.
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

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <iostream>
#include <lib/image/imageiopng.h>
#include <Eigen/Core>
#include <HDF5File.h>
#include <boost/program_options.hpp>

int main( int argc, char* argv[] )
{
	boost::program_options::options_description opt( "Option" );
	
	opt.add_options()
		( "help", "help" )
		( "input,i", boost::program_options::value<std::string>()->default_value( "input.exr" ), "input exr file name" )
		( "output,o", boost::program_options::value<std::string>()->default_value( "output.h5" ), "output hdf5 file name" )
		( "png,p", boost::program_options::value<std::string>()->default_value( "" ), "png file name (optional)" );
	
	boost::program_options::variables_map vm;
	try 
	{
		boost::program_options::store( boost::program_options::parse_command_line( argc, argv, opt ), vm );
	} 
	catch( const boost::program_options::error_with_option_name& e ) 
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
	boost::program_options::notify( vm );
	
	if( vm.count( "help" ) ) 
	{
		std::cout << opt << std::endl;
		return 0;
	}

	const std::string input_fn = vm["input"].as<std::string>();
	const std::string output_fn = vm["output"].as<std::string>();
	const std::string png_fn = vm["png"].as<std::string>();
	
	float* out = nullptr; // width * height * RGBA
	int width;
	int height;
	const char* err = nullptr;

	int ret = LoadEXR( &out, &width, &height, input_fn.c_str(), &err );

	if( ret != TINYEXR_SUCCESS ) 
	{
	  if( err ) 
		{
	    fprintf( stderr, "ERR : %s\n", err );
	    FreeEXRErrorMessage( err );
	  }
	} 
	else 
	{
		std::cout << "width: " << width << ", height: " << height << std::endl;
		
		if( png_fn != "" )
		{
			Image<float, 4> image;
			image.init( width, height );
			memcpy( image.getPtr(), out, sizeof( float ) * width * height * 4 );
			saveImagePng( png_fn, image );
		}
		
		Eigen::VectorXf data; data.resize( width * height * 4 );
		for( int i=0; i<width*height*4; i++ ) data(i) = out[i];
		HDF5File h5( output_fn, HDF5AccessType::READ_WRITE );
		Eigen::ArrayXi dims; dims.resize( 3 );
		dims(0) = height; dims(1) = width; dims(2) = 4;
		h5.write1DArrayAsTensor( "", "data", data, dims );
	
		free( out );
	}
	
	return 0;
}