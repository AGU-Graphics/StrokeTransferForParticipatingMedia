// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: framebufferobject.h
// Maintainer: Yonghao Yue
//
// Description:
// This file provides a simple wrapper for frame buffer object.
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

#ifndef __framebufferobject_h__
#define __framebufferobject_h__

class FrameBufferObject
{
	FrameBufferObject();
public:
	FrameBufferObject( const int in_width, const int in_height )
		: m_Width( in_width ), m_Height( in_height )
	{
		glGenTextures( 1, &m_ColorBuffer );
		glBindTexture( GL_TEXTURE_2D, m_ColorBuffer );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, nullptr );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glBindTexture( GL_TEXTURE_2D, 0 );
		
		glGenTextures( 1, &m_DepthBuffer );
		glBindTexture( GL_TEXTURE_2D, m_DepthBuffer );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_Width, m_Height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );	
		glBindTexture( GL_TEXTURE_2D, 0 );
		
		glGenFramebuffers( 1, &m_FrameBufferObject );
		glBindFramebuffer( GL_FRAMEBUFFER, m_FrameBufferObject );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorBuffer, 0 );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DepthBuffer, 0 );
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	}
	
	~FrameBufferObject()
	{
		glDeleteFramebuffers( 1, &m_FrameBufferObject );
		glDeleteTextures( 1, &m_ColorBuffer );
		glDeleteTextures( 1, &m_DepthBuffer );
	}
	
	void bind() const
	{
		glBindFramebuffer( GL_FRAMEBUFFER, m_FrameBufferObject );
	}
	
	void unbind() const
	{
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	}
	
	GLuint colorBuffer() const
	{
		return m_ColorBuffer;
	}
	
private:
	int m_Width;
	int m_Height;
	
	GLuint m_ColorBuffer;
	GLuint m_DepthBuffer;
	GLuint m_FrameBufferObject;
};

#endif