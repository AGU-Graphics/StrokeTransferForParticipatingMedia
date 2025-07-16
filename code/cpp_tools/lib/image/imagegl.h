// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: imagegl.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides helper functions for generating and updating textures from
// an image (object).
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


#ifndef image_gl_h
#define image_gl_h

#include <GLFW/glfw3.h>

template<typename T, int N>
GLuint generate_texture_from_image( const Image<T, N>& in_image ){ return 0; }

template<>
inline GLuint generate_texture_from_image( const Image<float, 3>& in_image )
{
  GLuint texture_idx;
  glGenTextures( 1, &texture_idx );
  glBindTexture( GL_TEXTURE_2D, texture_idx );

  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, in_image.getWidth(), in_image.getHeight(), 0, GL_RGB, GL_FLOAT, in_image.getPtr() );

  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

  return texture_idx;
}

template<>
inline GLuint generate_texture_from_image( const Image<float, 4>& in_image )
{
  GLuint texture_idx;
  glGenTextures( 1, &texture_idx );
  glBindTexture( GL_TEXTURE_2D, texture_idx );

  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, in_image.getWidth(), in_image.getHeight(), 0, GL_RGBA, GL_FLOAT, in_image.getPtr() );

  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

  return texture_idx;
}

template<typename T, int N>
void update_texture_from_image( const GLuint in_texture_idx, const Image<T, N>& in_image ) {}

template<>
inline void update_texture_from_image( const GLuint in_texture_idx, const Image<float, 4>& in_image )
{
  glBindTexture( GL_TEXTURE_2D, in_texture_idx );
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, in_image.getWidth(), in_image.getHeight(), GL_RGBA, GL_FLOAT, in_image.getPtr() );
}

#endif
