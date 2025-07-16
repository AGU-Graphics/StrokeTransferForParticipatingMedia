// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: gen_strokes_cli/src/main.cpp
// Maintainer: Yonghao Yue and Hideki Todo
//
// Description:
// This file implements the stroke rendering algorithm detailed in our 
// supplementary document Section 7.
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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Core>

#include <iostream>
#include <filesystem>

#include <lib/image/image.h>
#include <lib/image/imageioutil.h>
#include <lib/image/imageiopng.h>
#include <HDF5File.h>

#include <random>
#include <chrono>
#include <cmath>

#include <boost/program_options.hpp>

#include <lib/strokes/sortutil.h>
#include <lib/strokes/activeset.h>
#include <lib/strokes/anchor.h>
#include <lib/strokes/framebufferobject.h>
#include <lib/strokes/sampler.h>
#include <lib/strokes/stroke.h>

class StrokeRenderer
{
	StrokeRenderer();
public:
	StrokeRenderer( const int in_width, const int in_height )
		: m_Width( in_width ), m_Height( in_height ), m_Window( nullptr ), m_BufferBytes( nullptr )
	{
		m_Buffer.init( in_width, in_height );
		m_BufferBytes = (unsigned char*)malloc( sizeof(unsigned char) * in_width * in_height * 4 );
	}
		
	~StrokeRenderer()
	{
		free( m_BufferBytes );
		m_BufferBytes = nullptr;
		
		for( int i=0; i<m_GeneratedTextures.size(); i++ )
		{
			glDeleteTextures( 1, &m_GeneratedTextures[i] );
		}
		m_GeneratedTextures.clear();
	}
	
	bool initializeGL()
	{
	  if ( !glfwInit() )
	    return false;
		
	  glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
	  glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
	  glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
	  glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
		
		glfwWindowHint( GLFW_VISIBLE, 0 );
		glfwWindowHint( GLFW_DOUBLEBUFFER, GL_FALSE );
		
	  m_Window = glfwCreateWindow( m_Width, m_Height, "Stroke Renderer", NULL, NULL);
	  if ( !m_Window )
	  {
	    glfwTerminate();
	    return false;
	  }
		
	  glfwMakeContextCurrent( m_Window );
  
	  glewExperimental = GL_TRUE;
	  glewInit();
		
	  const GLubyte* renderer = glGetString( GL_RENDERER );
	  const GLubyte* version = glGetString( GL_VERSION );
	  std::cout << "Renderer: " << renderer << std::endl;
	  std::cout << "OpenGL version supported: " << version << std::endl;
		
	  compileDefaultShader();
			
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
		return true;
	}
	
	void compileDefaultShader()
	{
	  const char* vertex_shader =
	  "#version 400\n"
	  "in vec2 vp;\n"
		"layout(location = 1) in vec2 uv;\n"
	  "uniform mat4 MVP;\n"
		"out vec2 fragmentUV;\n"
	  "void main() {\n"
	  "  gl_Position = MVP * vec4(vp, 0.0, 1.0);\n"
		"  fragmentUV = uv;\n"
	  "}";
  
	  const char* fragment_shader =
	  "#version 400\n"
		"in vec2 fragmentUV;\n"
		"uniform vec4 strokeColor;\n"
		"uniform vec2 singleTexSize;\n"
		"uniform vec2 texSize;\n"
		"uniform int tex_id;\n"
		"uniform int mipmap_level;\n"
		"uniform float o_s;\n"
		"uniform float o_e;\n"
		"uniform sampler2D texLoc;\n"
	  "out vec4 finalColor;\n"
	  "void main() {\n"
		"  int w = 1 << mipmap_level;\n"
		"  float _left = ( w - 1.0 ) * singleTexSize.x / texSize.x;\n"
		"  float _right = ( 2.0 * w - 1.0 ) * singleTexSize.x / texSize.x;\n"
		"  float top = tex_id * singleTexSize.y / texSize.y;\n"
		"  float bottom = ( tex_id + 1 ) * singleTexSize.y / texSize.y;\n"
		"  float left = ( ( 1.0 - o_e ) * _left - o_s * _right ) / ( 1.0 - o_s - o_e );\n"
		"  float right = ( - o_e * _left + ( 1.0 - o_s ) * _right ) / ( 1.0 - o_s - o_e );\n"
		"  vec2 bb_min = vec2( left, top );\n"
		"  vec2 bb_max = vec2( right, bottom );\n"
		"  vec2 warped_uv = bb_min + ( bb_max - bb_min ) * fragmentUV;\n"
		"  finalColor = strokeColor * texture( texLoc, warped_uv );\n"
		"}";
  
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_Shader = glCreateProgram();
	  glAttachShader( m_Shader, fs );
	  glAttachShader( m_Shader, vs );
	  glLinkProgram( m_Shader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_Shader );
    const GLuint MatrixID = glGetUniformLocation( m_Shader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint texLoc = glGetUniformLocation( m_Shader, "texLoc" );
		glUniform1i( texLoc, 0 );
	}
	
	void compileVisibilityTestShader( const float in_TB, const float in_Talpha )
	{
	  const char* vertex_shader =
	  "#version 400\n"
	  "in vec2 vp;\n"
		"layout(location = 1) in vec2 uv;\n"
	  "uniform mat4 MVP;\n"
		"out vec2 fragmentUV;\n"
		"out vec2 xy_pos;\n"
	  "void main() {\n"
	  "  gl_Position = MVP * vec4(vp, 0.0, 1.0);\n"
		"  fragmentUV = uv;\n"
		"  xy_pos = vec2( ( gl_Position.x + 1.0 ) * 0.5, ( gl_Position.y + 1.0 ) * 0.5 );\n"
	  "}";
		
	  const char* fragment_shader =
	  "#version 400\n"
		"in vec2 fragmentUV;\n"
		"in vec2 xy_pos;\n"
		"uniform sampler2D texB;\n"
		"uniform sampler2D texLoc;\n"
		"uniform float TB;\n"
		"uniform float Talpha;\n"
		"uniform vec4 strokeColor;\n"
		"uniform vec2 singleTexSize;\n"
		"uniform vec2 texSize;\n"
		"uniform int tex_id;\n"
		"uniform int mipmap_level;\n"
		"uniform float o_s;\n"
		"uniform float o_e;\n"
	  "out vec4 finalColor;\n"
	  "void main() {\n"
		"  int w = 1 << mipmap_level;\n"
		"  float _left = ( w - 1.0 ) * singleTexSize.x / texSize.x;\n"
		"  float _right = ( 2.0 * w - 1.0 ) * singleTexSize.x / texSize.x;\n"
		"  float top = tex_id * singleTexSize.y / texSize.y;\n"
		"  float bottom = ( tex_id + 1 ) * singleTexSize.y / texSize.y;\n"
		"  float left = ( ( 1.0 - o_e ) * _left - o_s * _right ) / ( 1.0 - o_s - o_e );\n"
		"  float right = ( - o_e * _left + ( 1.0 - o_s ) * _right ) / ( 1.0 - o_s - o_e );\n"
		"  vec2 bb_min = vec2( left, top );\n"
		"  vec2 bb_max = vec2( right, bottom );\n"
		"  vec2 warped_uv = bb_min + ( bb_max - bb_min ) * fragmentUV;\n"
		"\n"
		"  vec4 B = texture( texB, xy_pos );\n"
			"  vec4 c = strokeColor * texture( texLoc, warped_uv );\n"
		"  if( c.w >= Talpha )\n"
		"  {\n"
		"    if( B.w <= TB )\n"
		"    {\n"
		"      discard;\n"
		"    }\n"
		"    else"
		"    {\n"
		"      finalColor = B;\n"
		"    }\n"
		"  }\n"
		"  else\n"
		"  {\n"
		"    discard;\n"
		"  }\n"
		"}";
		
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_VisibilityTestShader = glCreateProgram();
	  glAttachShader( m_VisibilityTestShader, fs );
	  glAttachShader( m_VisibilityTestShader, vs );
	  glLinkProgram( m_VisibilityTestShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_VisibilityTestShader );
    const GLuint MatrixID = glGetUniformLocation( m_VisibilityTestShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint texLoc = glGetUniformLocation( m_VisibilityTestShader, "texLoc" );
		glUniform1i( texLoc, 0 );
		const GLuint texB = glGetUniformLocation( m_VisibilityTestShader, "texB" );
		glUniform1i( texB, 1 );
		const GLuint TBLoc = glGetUniformLocation( m_VisibilityTestShader, "TB" );
		glUniform1f( TBLoc, in_TB );
		const GLuint TalphaLoc = glGetUniformLocation( m_VisibilityTestShader, "Talpha" );
		glUniform1f( TalphaLoc, in_Talpha );
	}
	
	bool finalize()
	{
		glfwTerminate();
		return true;
	}
	
	template<typename T, int N>
	GLuint setupStrokeTexture( const Image<T, N>& in_image )
	{
		GLuint texture = 0; 
	  glGenTextures( 1, &texture );
	  glBindTexture( GL_TEXTURE_2D, texture );
  
	  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, in_image.getWidth(), in_image.getHeight(), 0, GL_RGBA, GL_FLOAT, in_image.getPtr() );
  
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );	
		
		m_GeneratedTextures.push_back( texture );
		
		return texture;
	}
	
	void useTexture( GLuint in_Texture )
	{
		m_StrokeTexture = in_Texture;	
	}
	
	void clear()
	{
    glViewport( 0, 0, m_Width, m_Height );
		glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	}
	
	void clearB()
	{
    glViewport( 0, 0, m_Width, m_Height );
		glClearColor( 1.0, 1.0, 1.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	}
	
	void setTexAuxData( GLuint in_Shader, const TexAuxData& in_aux )
	{
		const GLuint singleTexSizeLoc = glGetUniformLocation( in_Shader, "singleTexSize" );
		float singleTexSizeVal[2] = { float( in_aux.single_w ), float( in_aux.single_h ) };
		glUniform2f( singleTexSizeLoc, singleTexSizeVal[0], singleTexSizeVal[1] );
		const GLuint texSizeLoc = glGetUniformLocation( in_Shader, "texSize" );
		float texSizeVal[2] = { float( in_aux.size(0) ), float( in_aux.size(1) ) };
		glUniform2f( texSizeLoc, texSizeVal[0], texSizeVal[1] );
	}
	
	GLuint defaultShader()
	{
		return m_Shader;
	}
	
	GLuint visibilityTestShader()
	{
		return m_VisibilityTestShader;
	}
	
	void drawTextureAsQuad( GLuint in_Shader )
	{
		const GLuint colorLoc = glGetUniformLocation( in_Shader, "strokeColor" );
		glUniform4f( colorLoc, 1.0, 1.0, 1.0, 1.0 );
		
		const GLuint texIdLoc = glGetUniformLocation( in_Shader, "tex_id" );
		glUniform1i( texIdLoc, 0 );
		const GLuint mipmapLevelLoc = glGetUniformLocation( in_Shader, "mipmap_level" );
		glUniform1i( mipmapLevelLoc, 0 );
		const GLuint o_sLoc = glGetUniformLocation( in_Shader, "o_s" );
		glUniform1f( o_sLoc, 0.0 );
		const GLuint o_eLoc = glGetUniformLocation( in_Shader, "o_e" );
		glUniform1f( o_eLoc, 0.0 );
		
		float points[] = {
			-1.0f, 1.0f,
			-1.0f, -1.0f,
			1.0f,  1.0f,
			-1.0f, -1.0f,
			1.0f, -1.0f,
			1.0f, 1.0f
		};
		
		float uvs[] = {
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f
		};
		
	  GLuint vbo = 0;
	  glGenBuffers( 1, &vbo );
	  glBindBuffer( GL_ARRAY_BUFFER, vbo );
	  glBufferData( GL_ARRAY_BUFFER, 3 * 2 * 2 * sizeof(float), points, GL_STATIC_DRAW );
		
	  GLuint uvbuffer = 0;
	  glGenBuffers(1, &uvbuffer);
	  glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	  glBufferData(GL_ARRAY_BUFFER, 3 * 2 * 2 * sizeof(float), uvs, GL_STATIC_DRAW);
  
	  GLuint vao = 0;
	  glGenVertexArrays( 1, &vao );
	  glBindVertexArray( vao );
	  glEnableVertexAttribArray(0);
	  glBindBuffer( GL_ARRAY_BUFFER, vbo );
	  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, NULL );
		
	  glEnableVertexAttribArray(1);
	  glBindBuffer( GL_ARRAY_BUFFER, uvbuffer );
	  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, NULL );
  	
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, m_StrokeTexture );
		
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * 2 );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteVertexArrays( 1, &vao );
	}
	
	void draw( GLuint in_Shader, const Stroke& in_stroke, const Eigen::Vector2i& in_info_int, const Eigen::Vector2f& in_info_float )
	{
		const GLuint colorLoc = glGetUniformLocation( in_Shader, "strokeColor" );
		float color[4]; in_stroke.color( color );
		glUniform4f( colorLoc, color[0], color[1], color[2], color[3] );
		
		const GLuint texIdLoc = glGetUniformLocation( in_Shader, "tex_id" );
		glUniform1i( texIdLoc, in_info_int(0) );
		const GLuint mipmapLevelLoc = glGetUniformLocation( in_Shader, "mipmap_level" );
		glUniform1i( mipmapLevelLoc, in_info_int(1) );
		const GLuint o_sLoc = glGetUniformLocation( in_Shader, "o_s" );
		glUniform1f( o_sLoc, in_info_float(0) );
		const GLuint o_eLoc = glGetUniformLocation( in_Shader, "o_e" );
		glUniform1f( o_eLoc, in_info_float(1) );
		
	  GLuint vbo = 0;
	  glGenBuffers( 1, &vbo );
	  glBindBuffer( GL_ARRAY_BUFFER, vbo );
	  glBufferData( GL_ARRAY_BUFFER, 3 * 2 * in_stroke.numTris() * sizeof(float), in_stroke.vertices(), GL_STATIC_DRAW );
		
	  GLuint uvbuffer = 0;
	  glGenBuffers(1, &uvbuffer);
	  glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	  glBufferData(GL_ARRAY_BUFFER, 3 * 2 * in_stroke.numTris() * sizeof(float), in_stroke.uvs(), GL_STATIC_DRAW);
  
	  GLuint vao = 0;
	  glGenVertexArrays( 1, &vao );
	  glBindVertexArray( vao );
	  glEnableVertexAttribArray(0);
	  glBindBuffer( GL_ARRAY_BUFFER, vbo );
	  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, NULL );
		
	  glEnableVertexAttribArray(1);
	  glBindBuffer( GL_ARRAY_BUFFER, uvbuffer );
	  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, NULL );
  	
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, m_StrokeTexture );
		
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * in_stroke.numTris() );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteVertexArrays( 1, &vao );
	}
	
	void draw( GLuint in_Shader, const StrokeData& in_strokes )
	{
		for( int i=0; i<in_strokes.strokes.size(); i++ )
		{
			draw( in_Shader, in_strokes.strokes[i], in_strokes.tex_info_int[i], in_strokes.tex_info_float[i] );
		}
	}
	
	void readBuffer( Image<float, 4>& io_buffer )
	{
		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glReadPixels( 0, 0, m_Width, m_Height, GL_RGBA, GL_FLOAT, io_buffer.getPtr() );
	}
	
	void readBufferBytes( unsigned char* io_buffer )
	{
		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glReadPixels( 0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, io_buffer );
	}
	
	void readSubBuffer( Image<float, 4>& io_buffer, int in_x, int in_y, int in_w, int in_h )
	{
		const int buffer_width = io_buffer.getWidth();
		
		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glReadPixels( in_x, in_y, in_w, in_h, GL_RGBA, GL_FLOAT, m_Buffer.getPtr() );
				
		for( int j=0; j<in_h; j++ )
		{
			float* p_src = &( m_Buffer.getPtr()[ ( j * in_w ) * 4 ] );
			float* p_dst = &( io_buffer.getPtr()[ ( (j+in_y)*buffer_width+(in_x) ) * 4 ] );
			
			for( int i=0; i<in_w*4; i++ )
			{
				*p_dst++=*p_src++;
			}
		}
	}
	
	void readSubBufferBytes( unsigned char* io_buffer, int in_buffer_width, int in_x, int in_y, int in_w, int in_h )
	{		
		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glReadPixels( in_x, in_y, in_w, in_h, GL_RGBA, GL_UNSIGNED_BYTE, m_BufferBytes );
				
		for( int j=0; j<in_h; j++ )
		{
			unsigned char* p_src = &( m_BufferBytes[ ( j * in_w ) * 4 ] );
			unsigned char* p_dst = &( io_buffer[ ( (j+in_y)*in_buffer_width+(in_x) ) * 4 ] );
			
			for( int i=0; i<in_w*4; i++ )
			{
				*p_dst++=*p_src++;
			}
		}
	}
		
private:
	int m_Width;
	int m_Height;
	GLFWwindow* m_Window;
	GLuint m_Shader;
	GLuint m_StrokeTexture;
	glm::mat4 m_MVP;
	
	GLuint m_VisibilityTestShader;
	
	Image<float, 4> m_Buffer;
	unsigned char* m_BufferBytes;
	
	std::vector<GLuint> m_GeneratedTextures;
};

void advectPointsTVDRK3( std::vector<Eigen::Vector2f>& io_pos, const float in_dt_per_step, const int in_num_sub_steps, const VelocitySampler* in_velocity_sampler )
{
  // https://www.ams.org/journals/mcom/1998-67-221/S0025-5718-98-00913-2/S0025-5718-98-00913-2.pdf
  // Third order TVD-RK:
  // du/dt = L(u)
  // u^(1) = u^n + dt L(u^n)
  // u^(2) = 3/4 u^n + 1/4 u^(1) + 1/4 dt L(u^(1))
  // u^n+1 = 1/3 u^n + 2/3 u^(2) + 2/3 dt L(u^(2))
	  
  const float dt = in_dt_per_step / in_num_sub_steps;

	for( int i=0; i<io_pos.size(); i++ )
	{
	  float _pts_x = io_pos[i](0);
	  float _pts_y = io_pos[i](1);
		
		for( int k=0; k<in_num_sub_steps; k++ )
		{
			float _u[2];
			in_velocity_sampler->sample( _pts_x, _pts_y, _u ); 
	    const float pts_x_1 = _pts_x + dt * _u[0];
	    const float pts_y_1 = _pts_y + dt * _u[1];
			float _u_1[2];
			in_velocity_sampler->sample( pts_x_1, pts_y_1, _u_1 );
	    const float pts_x_2 = _pts_x * 3.0 / 4.0 + pts_x_1 / 4.0 + dt * _u_1[0] / 4.0;
	    const float pts_y_2 = _pts_y * 3.0 / 4.0 + pts_y_1 / 4.0 + dt * _u_1[1] / 4.0;
	    float _u_2[2];
			in_velocity_sampler->sample( pts_x_2, pts_y_2, _u_2 );
	    _pts_x = _pts_x / 3.0 + pts_x_2 * 2.0 / 3.0 + dt * _u_2[0] * 2.0 / 3.0;
	    _pts_y = _pts_y / 3.0 + pts_y_2 * 2.0 / 3.0 + dt * _u_2[1] * 2.0 / 3.0;			
		}
		
		io_pos[i](0) = _pts_x;
		io_pos[i](1) = _pts_y;
	}
}

float buildCenterLineTVDRK3( std::vector<Eigen::Vector2f>& out_center_line, const Eigen::Vector2f& in_c, 
	const Sampler* in_length_sampler, const OrientationSampler* in_orientation_sampler, const Sampler* in_mask_sampler,
	const float in_anchor_label, const Sampler* in_region_label_sampler,
	const float in_orientation_factor, const float in_random_offset, 
	const float in_random_offset_factor = 1.0, const float in_length_factor = 1.0, const float in_step_length = 0.01, const float in_step_length_accuracy = 0.1, 
	const float in_default_dt = 1.0, const float in_dt_cap = 1.0e4, const float in_dt_accuracy = 0.001, bool in_verbose = false )
{
  // https://www.ams.org/journals/mcom/1998-67-221/S0025-5718-98-00913-2/S0025-5718-98-00913-2.pdf
  // Third order TVD-RK:
  // du/dt = L(u)
  // u^(1) = u^n + dt L(u^n)
  // u^(2) = 3/4 u^n + 1/4 u^(1) + 1/4 dt L(u^(1))
  // u^n+1 = 1/3 u^n + 2/3 u^(2) + 2/3 dt L(u^(2))

	out_center_line.clear();
  out_center_line.push_back( in_c );
	
  float _pts_x = in_c(0);
  float _pts_y = in_c(1);
    
  const float theta = in_random_offset * M_PI * in_random_offset_factor / 180.0;
  const float sin_theta = sin( theta );
  const float cos_theta = cos( theta );
    
  float length = 0.0;
  
  float min_dt = 0.0;
  float max_dt = in_default_dt;
  float mid_dt = in_default_dt;
  int n = 1;
	
	float sampler_length = 0.0;
	in_length_sampler->sample( _pts_x, _pts_y, &sampler_length );
  sampler_length *= in_length_factor;
    
  float target_length = sampler_length;
	
	while(1)
	{	
		float __u[2];
		in_orientation_sampler->sample( _pts_x, _pts_y, __u ); __u[0] *= in_orientation_factor; __u[1] *= in_orientation_factor;
    float _ux = cos_theta * __u[0] + sin_theta * __u[1];
    float _uy = -sin_theta * __u[0] + cos_theta * __u[1];
		
		if( !isfinite( _ux ) || !isfinite( _uy ) )
		{
			if( in_verbose ) std::cout << "****** WARNING! Invalid velocity observed in buildCenterLineTVDRK3! Skipping the rest of computation..." << std::endl;
			break;
		}
		
		if( _ux * _ux + _uy * _uy > 1.05 )
		{
			if( in_verbose ) std::cout << "****** WARNING! Observed unnormalized velocity in buildCenterLineTVDRK3! Performing normalization..." << std::endl;
			const float __len = sqrt( _ux * _ux + _uy * _uy );
			_ux = _ux / __len;
			_uy = _uy / __len;
		}
		
    const float pts_x_1 = _pts_x + mid_dt * _ux;
    const float pts_y_1 = _pts_y + mid_dt * _uy;
		float __u_1[2];
		in_orientation_sampler->sample( pts_x_1, pts_y_1, __u_1 ); __u_1[0] *= in_orientation_factor; __u_1[1] *= in_orientation_factor;
		float _ux_1 = cos_theta * __u_1[0] + sin_theta * __u_1[1];
    float _uy_1 = -sin_theta * __u_1[0] + cos_theta * __u_1[1];
		
		if( !isfinite( _ux_1 ) || !isfinite( _uy_1 ) )
		{
			if( in_verbose ) std::cout << "****** WARNING! Invalid velocity observed in buildCenterLineTVDRK3! Skipping the rest of computation..." << std::endl;
			break;
		}
		
		if( _ux_1 * _ux_1 + _uy_1 * _uy_1 > 1.05 )
		{
			if( in_verbose ) std::cout << "****** WARNING! Observed unnormalized velocity in buildCenterLineTVDRK3! Performing normalization..." << std::endl;
			const float __len = sqrt( _ux_1 * _ux_1 + _uy_1 * _uy_1 );
			_ux_1 = _ux_1 / __len;
			_uy_1 = _uy_1 / __len;
		}
		
    const float pts_x_2 = _pts_x * 3.0 / 4.0 + pts_x_1 / 4.0 + mid_dt * _ux_1 / 4.0;
    const float pts_y_2 = _pts_y * 3.0 / 4.0 + pts_y_1 / 4.0 + mid_dt * _uy_1 / 4.0;
    float __u_2[2];
		in_orientation_sampler->sample( pts_x_2, pts_y_2, __u_2 ); __u_2[0] *= in_orientation_factor; __u_2[1] *= in_orientation_factor;
		float _ux_2 = cos_theta * __u_2[0] + sin_theta * __u_2[1];
    float _uy_2 = -sin_theta * __u_2[0] + cos_theta * __u_2[1];
		
		if( !isfinite( _ux_2 ) || !isfinite( _uy_2 ) )
		{
			if( in_verbose ) std::cout << "****** WARNING! Invalid velocity observed in buildCenterLineTVDRK3! Skipping the rest of computation..." << std::endl;
			break;
		}
		
		if( _ux_2 * _ux_2 + _uy_2 * _uy_2 > 1.05 )
		{
			if( in_verbose ) std::cout << "****** WARNING! Observed unnormalized velocity in buildCenterLineTVDRK3! Performing normalization..." << std::endl;
			const float __len = sqrt( _ux_2 * _ux_2 + _uy_2 * _uy_2 );
			_ux_2 = _ux_2 / __len;
			_uy_2 = _uy_2 / __len;
		}
		
    const float _pts_x_next = _pts_x / 3.0 + pts_x_2 * 2.0 / 3.0 + mid_dt * _ux_2 * 2.0 / 3.0;
    const float _pts_y_next = _pts_y / 3.0 + pts_y_2 * 2.0 / 3.0 + mid_dt * _uy_2 * 2.0 / 3.0;
    
    const float _length = sqrt( ( _pts_x_next - _pts_x ) * ( _pts_x_next - _pts_x ) + ( _pts_y_next - _pts_y ) * ( _pts_y_next - _pts_y ) );
            
    if( fabs( _length - in_step_length ) < in_step_length * in_step_length_accuracy || ( fabs( _length - in_step_length ) < in_step_length * 0.8 && fabs( max_dt - min_dt ) < ( max_dt + min_dt ) * 0.5 * in_dt_accuracy ) )
		{
      length += _length;
      _pts_x = _pts_x_next;
      _pts_y = _pts_y_next;
			
      out_center_line.emplace_back( _pts_x, _pts_y );

			float mask;
      in_mask_sampler->sample( _pts_x, _pts_y, &mask );
      if( mask < 0.5 )
        break;
			
			float label;
			in_region_label_sampler->sample( _pts_x, _pts_y, &label );
			if( fabs( in_anchor_label - label ) > 0.5 )
				break;
      
			in_length_sampler->sample( _pts_x, _pts_y, &sampler_length );
      sampler_length *= in_length_factor;
      
      target_length = ( target_length * n + sampler_length ) / ( n + 1.0 );
      n += 1;

      min_dt = 0.0;
      max_dt = in_default_dt;
      mid_dt = in_default_dt;
    
      if( length >= target_length )
        break;
		}
		else if( fabs( max_dt - min_dt ) < ( max_dt + min_dt ) * 0.5 * in_dt_accuracy )
		{
			if( in_verbose ) 
			{
				std::cout << "**********************************************************************************************" << std::endl;
      	std::cout << "  buildCenterLineTVDRK3() failed to find a line segment within the given tolerance." << std::endl;
				std::cout << "  This may happen due to abrupt change in the given velocity field." << std::endl;
				std::cout << "  The built centerline is discarded to prevent possible artifacts in the stroke rendering..." << std::endl;
				std::cout << "**********************************************************************************************" << std::endl;
			}
					
			out_center_line.clear();
			out_center_line.push_back( in_c );
			length = 0.0;
			
			break;
		}
		else if( _length < in_step_length && mid_dt >= max_dt )
		{
      min_dt = mid_dt;
      mid_dt *= 2.0;
      max_dt = mid_dt;
		}
		else if( _length < in_step_length )
		{
      min_dt = mid_dt;
      mid_dt = ( mid_dt + max_dt ) * 0.5;
		}
		else if( mid_dt <= min_dt )
		{
      max_dt = mid_dt;
      mid_dt *= 0.5;
      min_dt = mid_dt;
		}
    else
		{
      max_dt = mid_dt;
      mid_dt = ( min_dt + mid_dt ) * 0.5;
		}
 
    if( min_dt >= in_dt_cap )
		{
      // we should skip the resulting vertex, as _length is likely zero
      break;
		}
	}     
  
  return length;
}

void sort_strokes_by_luminance( AnchorData& io_anchors, StrokeData& io_strokes )
{
	std::function<float(const Stroke&)> _f = luminance;
	std::vector<SortItem> list;
	buildSortItemList( io_strokes.strokes, _f, list );
	std::sort( list.begin(), list.end() );
	io_anchors.rearrange( list );
	io_strokes.rearrange( list );
}

void sort_strokes_by_index( AnchorData& io_anchors, StrokeData& io_strokes )
{
	std::function<float(const int&)> _f = sindex;
	std::vector<SortItem> list;
	buildSortItemList( io_strokes.indices, _f, list );
	std::sort( list.begin(), list.end() );
	io_anchors.rearrange( list );
	io_strokes.rearrange( list );
}

void sort_strokes_frame_set_by_luminance( AnchorData& io_merged_anchors, StrokeData& io_merged_strokes, AnchorData& io_added_anchors, StrokeData& io_added_strokes )
{
	std::function<float(const Stroke&)> _f = luminance;
	std::vector<SortItem> list;
	buildSortItemList( io_added_strokes.strokes, _f, list );
	std::sort( list.begin(), list.end() );
	io_added_anchors.rearrange( list );
	io_added_strokes.rearrange( list );
	
	io_merged_anchors.append( io_added_anchors );
	io_merged_strokes.append( io_added_strokes );
}

void sort_strokes_frame_set_by_index( AnchorData& io_merged_anchors, StrokeData& io_merged_strokes, AnchorData& io_added_anchors, StrokeData& io_added_strokes )
{
	std::function<float(const int&)> _f = sindex;
	std::vector<SortItem> list;
	buildSortItemList( io_added_strokes.indices, _f, list );
	std::sort( list.begin(), list.end() );
	io_added_anchors.rearrange( list );
	io_added_strokes.rearrange( list );
	
	io_merged_anchors.append( io_added_anchors );
	io_merged_strokes.append( io_added_strokes );
}

void all_sort_strokes_frame_set_by_luminance( AnchorData& io_merged_anchors, StrokeData& io_merged_strokes, AnchorData& io_added_anchors, StrokeData& io_added_strokes )
{
	io_merged_anchors.append( io_added_anchors );
	io_merged_strokes.append( io_added_strokes );
		
	std::function<float(const Stroke&)> _f = luminance;
	std::vector<SortItem> list;
	buildSortItemList( io_merged_strokes.strokes, _f, list );
	std::sort( list.begin(), list.end() );
	io_merged_anchors.rearrange( list );
	io_merged_strokes.rearrange( list );
}

void all_sort_strokes_frame_set_by_index( AnchorData& io_merged_anchors, StrokeData& io_merged_strokes, AnchorData& io_added_anchors, StrokeData& io_added_strokes )
{
	io_merged_anchors.append( io_added_anchors );
	io_merged_strokes.append( io_added_strokes );
		
	std::function<float(const int&)> _f = sindex;
	std::vector<SortItem> list;
	buildSortItemList( io_merged_strokes.indices, _f, list );
	std::sort( list.begin(), list.end() );
	io_merged_anchors.rearrange( list );
	io_merged_strokes.rearrange( list );
}

void updateAnchorPointsAndStrokes( StrokeData& out_strokes, StrokeRenderer* io_renderer, unsigned char* io_image_bytes, HierarchicalActiveSet* io_has, std::mt19937& io_mt, 
	const AnchorData& in_anchors, AnchorData& out_anchors, const TexAuxData& in_tex_aux_data,
	const Sampler* in_color_sampler, const Sampler* in_length_sampler, const Sampler* in_width_sampler, const OrientationSampler* in_orientation_sampler, const Sampler* in_mask_sampler,
	const Sampler* in_region_label_sampler,
	const int in_frame_idx, const int in_width, const int in_height, 
  const float in_random_offset_factor, const float in_length_factor, const float in_width_factor, const float in_step_length, const float in_step_length_accuracy )
{
	out_strokes.clear();
	out_anchors.clear();
	
	std::uniform_real_distribution<> dist( 0.0, 1.0 );
	
	for( int i=0; i<in_anchors.pos.size(); i++ )
	{
		std::vector<Eigen::Vector2f> center_line;
		
		float anchor_guide_label;
		in_region_label_sampler->sample( in_anchors.pos[i].x(), in_anchors.pos[i].y(), &anchor_guide_label );
	
		std::vector<Eigen::Vector2f> center_line_plus;
		const float length_plus = buildCenterLineTVDRK3( center_line_plus, in_anchors.pos[i], in_length_sampler, in_orientation_sampler, in_mask_sampler, anchor_guide_label, in_region_label_sampler, 
			1.0, in_anchors.random_numbers[i](0), in_random_offset_factor, 0.5 * in_length_factor * ( 1.0 - in_anchors.random_numbers[i](1) ), in_step_length, in_step_length_accuracy );
	
		std::vector<Eigen::Vector2f> center_line_minus;
		const float length_minus = buildCenterLineTVDRK3( center_line_minus, in_anchors.pos[i], in_length_sampler, in_orientation_sampler, in_mask_sampler, anchor_guide_label, in_region_label_sampler, 
			-1.0, in_anchors.random_numbers[i](0), in_random_offset_factor, 0.5 * in_length_factor * ( 1.0 - in_anchors.random_numbers[i](1) ), in_step_length, in_step_length_accuracy );			
	
		if( center_line_plus.size() + center_line_minus.size() - 1 <= 1 )
			continue;
		
		out_anchors.appendFromOther( in_anchors, i );
	
		center_line.resize( center_line_plus.size() + center_line_minus.size() - 1 );
		for( int i=1; i<center_line_minus.size(); i++ )
			center_line[i-1] = center_line_minus[ center_line_minus.size() - i ];
		for( int i=0; i<center_line_plus.size(); i++ )
			center_line[center_line_minus.size()-1+i] = center_line_plus[i];
	
		const float length = length_minus + length_plus;
	
		float color[4];
		in_color_sampler->sample( in_anchors.pos[i].x(), in_anchors.pos[i].y(), color );
	
		float width;
		in_width_sampler->sample( in_anchors.pos[i].x(), in_anchors.pos[i].y(), &width );
	
	  const float s_width = width * in_width_factor * ( 1.0 - in_anchors.random_numbers[i](2) );
	  const float s_length = length;
	  const float s_ratio = s_length / s_width;
	  const int mip_map_level = std::max<int>( 0, std::min<int>( int( floor( log2( s_ratio / in_tex_aux_data.min_tex_ratio ) + 0.5 ) ), in_tex_aux_data.length_mipmap_level - 1 ) );
	  const int tex_id = std::max<int>( 0, std::min<int>( int( floor( in_anchors.random_numbers[i](3) * in_tex_aux_data.num_textures ) ), in_tex_aux_data.num_textures - 1 ) );
	
		out_strokes.strokes.emplace_back( center_line, s_width, color );
		out_strokes.indices.emplace_back( in_anchors.indices[i] );
    out_strokes.tex_info_int.emplace_back( tex_id, mip_map_level );
		out_strokes.tex_info_float.emplace_back( in_anchors.random_numbers[i](4), in_anchors.random_numbers[i](5) );
		
		io_renderer->draw( io_renderer->defaultShader(), out_strokes.strokes[out_strokes.strokes.size()-1], out_strokes.tex_info_int[out_strokes.tex_info_int.size()-1], out_strokes.tex_info_float[out_strokes.tex_info_float.size()-1] );
	}
	
	io_renderer->readBufferBytes( io_image_bytes );
	io_has->setActiveSetFromRGBABufferBytes( io_image_bytes, in_width, in_height, 0.5 );
}

int generateAnchorPointsAndStrokes( AnchorData& out_anchors, StrokeData& out_strokes, StrokeRenderer* io_renderer, unsigned char* io_image_bytes, HierarchicalActiveSet* io_has, std::mt19937& io_mt, 
	const std::vector<float>& in_max_random, const std::vector<bool>& in_random_pm, 
	const TexAuxData& in_tex_aux_data,
	const Sampler* in_color_sampler, const Sampler* in_length_sampler, const Sampler* in_width_sampler, const OrientationSampler* in_orientation_sampler, const Sampler* in_mask_sampler,
	const Sampler* in_region_label_sampler,
	const int in_frame_idx, const int in_width, const int in_height, 
	const float in_random_offset_factor, const float in_length_factor, const float in_width_factor, const int in_next_idx, const float in_step_length, const float in_step_length_accuracy, const int in_consecutive_failure_max )
{
	out_anchors.clear();
	out_strokes.clear();
	std::uniform_real_distribution<> dist( 0.0, 1.0 );
	
	const int max_res = std::max<int>( in_width, in_height );

	constexpr int num_random_numbers = 6;	
	if( in_max_random.size() != num_random_numbers || in_random_pm.size() != num_random_numbers )
	{
		std::cout << "Please update generateAnchorPointsAndStrokes() so that it will generate " << in_max_random.size() 
			<< " random numbers per anchor point (currently it generates 6)." << std::endl;
		exit(-1);
	}
	
	Eigen::Matrix<float,num_random_numbers,1> random_numbers;
	
	int anchor_point_index = in_next_idx;
	
	int num_failed = 0;
	int prev_unoccupied = -1;
	
	while(1)
	{
		int xi, yi;
		int num_unoccupied = io_has->findUnoccupied( dist( io_mt ), xi, yi );
		if( num_unoccupied <= 0 ) break;
		
		if( num_unoccupied == prev_unoccupied ) num_failed++;
		else num_failed = 0;

		prev_unoccupied = num_unoccupied;
		
		if( num_failed >= in_consecutive_failure_max ) break;
		
		for( int i=0; i<num_random_numbers; i++ )
			random_numbers(i) = in_random_pm[i] ? ( 2.0 * dist( io_mt ) - 1.0 ) * in_max_random[i] : dist( io_mt ) * in_max_random[i];
		
		Eigen::Vector2f _c{ float( xi ) / max_res, float( in_height - yi ) / max_res }; //[0, 1]
		
		float anchor_guide_label;
		in_region_label_sampler->sample( _c.x(), _c.y(), &anchor_guide_label );
		
		std::vector<Eigen::Vector2f> center_line_plus;
		const float length_plus = buildCenterLineTVDRK3( center_line_plus, _c, in_length_sampler, in_orientation_sampler, in_mask_sampler, anchor_guide_label, in_region_label_sampler,
			1.0, random_numbers(0), in_random_offset_factor, 0.5 * in_length_factor * ( 1.0 - random_numbers(1) ), in_step_length, in_step_length_accuracy );
		
		std::vector<Eigen::Vector2f> center_line_minus;
		const float length_minus = buildCenterLineTVDRK3( center_line_minus, _c, in_length_sampler, in_orientation_sampler, in_mask_sampler, anchor_guide_label, in_region_label_sampler,
			-1.0, random_numbers(0), in_random_offset_factor, 0.5 * in_length_factor * ( 1.0 - random_numbers(1) ), in_step_length, in_step_length_accuracy );			
		
		if( center_line_plus.size() + center_line_minus.size() - 1 <= 1 )
			continue;
		
		out_anchors.pos.push_back( _c );
		out_anchors.born_time.push_back( in_frame_idx );
		out_anchors.indices.push_back( anchor_point_index );
		out_anchors.random_numbers.push_back( random_numbers );
		
		std::vector<Eigen::Vector2f> center_line;
		center_line.resize( center_line_plus.size() + center_line_minus.size() - 1 );
		for( int i=1; i<center_line_minus.size(); i++ )
			center_line[i-1] = center_line_minus[ center_line_minus.size() - i ];
		for( int i=0; i<center_line_plus.size(); i++ )
			center_line[center_line_minus.size()-1+i] = center_line_plus[i];
		
		const float length = length_minus + length_plus;
		
		float color[4];
		in_color_sampler->sample( _c.x(), _c.y(), color );
		
		float width;
		in_width_sampler->sample( _c.x(), _c.y(), &width );
		
    const float s_width = width * in_width_factor * ( 1.0 - random_numbers(2) );
    const float s_length = length;
    const float s_ratio = s_length / s_width;
    const int mip_map_level = std::max<int>( 0, std::min<int>( int( floor( log2( s_ratio / in_tex_aux_data.min_tex_ratio ) + 0.5 ) ), in_tex_aux_data.length_mipmap_level - 1 ) );
    const int tex_id = std::max<int>( 0, std::min<int>( int( floor( random_numbers(3) * in_tex_aux_data.num_textures ) ), in_tex_aux_data.num_textures - 1 ) );
		
		out_strokes.strokes.emplace_back( center_line, s_width, color );
		
    out_strokes.tex_info_int.emplace_back( tex_id, mip_map_level );
		out_strokes.tex_info_float.emplace_back( random_numbers(4), random_numbers(5) );
		
		io_renderer->draw( io_renderer->defaultShader(), out_strokes.strokes[out_strokes.strokes.size()-1], out_strokes.tex_info_int[out_strokes.tex_info_int.size()-1], out_strokes.tex_info_float[out_strokes.tex_info_float.size()-1] );
		
		const Eigen::Vector2f BBMin = out_strokes.strokes[out_strokes.strokes.size()-1].BBMin();
		const Eigen::Vector2f BBMax = out_strokes.strokes[out_strokes.strokes.size()-1].BBMax();
		
		int st_x = std::max<int>( 0, std::min<int>( in_width - 1, int( floor( max_res * BBMin(0) ) ) ) );
		int st_y = std::max<int>( 0, std::min<int>( in_height - 1, int( floor( in_height - max_res * BBMax(1) ) ) ) );
		int st_r = std::max<int>( 0, std::min<int>( in_width - 1, int( ceil( max_res * BBMax(0) ) ) ) );
		int st_b = std::max<int>( 0, std::min<int>( in_height - 1, int( ceil( in_height - max_res * BBMin(1) ) ) ) );
		int st_width = st_r - st_x + 1;
		int st_height = st_b - st_y + 1;
		
    out_strokes.indices.push_back( anchor_point_index );
		
		io_renderer->readSubBufferBytes( io_image_bytes, in_width, st_x, st_y, st_width, st_height );
		io_has->updateActiveSetFromRGBABufferBytes( io_image_bytes, in_width, st_x, st_y, st_width, st_height, 0.5 );
		
		anchor_point_index += 1;
	}
	
	return anchor_point_index;
}

int main( int argc, char* argv[] )
{
	boost::program_options::options_description opt( "Option" );
	opt.add_options()
		("help", "help")
		("width,w", boost::program_options::value<int>()->default_value(1024), "width")
		("height,h", boost::program_options::value<int>()->default_value(1024), "height")
		("vx_filename_template", boost::program_options::value<std::string>()->default_value("velocty_u_%03d.h5"), "vx_filename_template" )
		("vy_filename_template", boost::program_options::value<std::string>()->default_value("velocty_v_%03d.h5"), "vy_filename_template" )
		("frame_start", boost::program_options::value<int>()->default_value(140), "frame_start" )
		("frame_end", boost::program_options::value<int>()->default_value(141), "frame_end" )
		("frame_skip", boost::program_options::value<int>()->default_value(1), "frame_skip" )
		("angular_random_offset_deg", boost::program_options::value<float>()->default_value(5.0), "angular_random_offset_deg" )
		("random_offset_factor", boost::program_options::value<float>()->default_value(1.0), "random_offset_factor" )
		("length_factor", boost::program_options::value<float>()->default_value(1.0), "length_factor" )
		("width_factor", boost::program_options::value<float>()->default_value(1.0), "width_factor" )
		("length_random_factor_relative", boost::program_options::value<float>()->default_value(0.0), "length_random_factor_relative" )
		("width_random_factor_relative", boost::program_options::value<float>()->default_value(0.0), "width_random_factor_relative" )
		("stroke_step_length", boost::program_options::value<float>()->default_value(1.0/256.0), "stroke_step_length" )
		("stroke_step_length_accuracy", boost::program_options::value<float>()->default_value(0.1), "stroke_step_length_accuracy" )
		("consecutive_failure_max", boost::program_options::value<int>()->default_value(100), "consecutive_failure_max" )
		("texture_start_max_random_offset", boost::program_options::value<float>()->default_value(0.0), "texture_start_max_random_offset" )
		("texture_end_max_random_offset", boost::program_options::value<float>()->default_value(0.0), "texture_end_max_random_offset" )
		("sort_type", boost::program_options::value<std::string>()->default_value("add_sort_index"), "sort_type" )
		("clip_with_undercoat_alpha", boost::program_options::value<int>()->default_value(0), "clip_with_undercoat_alpha" )
		("texture_filename", boost::program_options::value<std::string>()->default_value("../../textures/texture.png"), "texture_filename" )
		("texture_for_active_set_for_new_stroke_filename", boost::program_options::value<std::string>()->default_value("../../textures/texture_for_new.png"), "texture_for_active_set_for_new_stroke_filename" )
		("texture_for_active_set_for_existing_stroke_filename", boost::program_options::value<std::string>()->default_value("../../textures/texture_for_existing.png"), "texture_for_active_set_for_existing_stroke_filename" )
		("num_textures", boost::program_options::value<int>()->default_value(1), "num_textures" )
		("texture_length_mipmap_level", boost::program_options::value<int>()->default_value(1), "texture_length_mipmap_level" )
		("out_anchor_filename_template", boost::program_options::value<std::string>()->default_value("./anchor/anchor_%d.hdf5"), "out_anchor_filename_template" )
		("out_stroke_filename_template", boost::program_options::value<std::string>()->default_value("stroke/stroke_%03d.png"), "out_stroke_filename_template" )
		("out_stroke_data_filename_template", boost::program_options::value<std::string>()->default_value("stroke_data/stroke_%03d.h5"), "out_stroke_data_filename_template" )
		("orientation_filename_template", boost::program_options::value<std::string>()->default_value("smooth_orientation/smooth_orientation_%03d.hdf5"), "orientation_filename_template" )
		("undercoat_filename_template", boost::program_options::value<std::string>()->default_value("color/color_%03d.png"), "undercoat_filename_template" )
		("color_filename_template", boost::program_options::value<std::string>()->default_value("color/color_%03d.hdf5"), "color_filename_template" )
		("width_filename_template", boost::program_options::value<std::string>()->default_value("width/width_%03d.hdf5"), "width_filename_template" )
		("length_filename_template", boost::program_options::value<std::string>()->default_value("length/length_%03d.hdf5"), "length_filename_template" )
		("frame_rate", boost::program_options::value<float>()->default_value(24.0), "frame_rate" )
		("mask_file_template", boost::program_options::value<std::string>()->default_value(""), "mask_file_template" )
		("active_set_file_template", boost::program_options::value<std::string>()->default_value(""), "active_set_file_template" )
		("region_label_template", boost::program_options::value<std::string>()->default_value(""), "region_label_template" )
		("remove_hidden_strokes", boost::program_options::value<int>()->default_value(1), "remove_hidden_strokes" )
		("remove_hidden_strokes_thr_contribution", boost::program_options::value<float>()->default_value(1.0/256.0), "if all pixels of a stroke has contribution (discounted by other strokes in front of it) below this threshold, it will be dictated as invisible" )
		("remove_hidden_strokes_thr_alpha", boost::program_options::value<float>()->default_value(1.0/256.0), "pixels of a stroke subject to the above dictation are those with alpha values above this threshold" )
		("log_file_name", boost::program_options::value<std::string>()->default_value("stroke/log.txt"), "log_file_name" )
		("resume_frame_idx", boost::program_options::value<int>()->default_value(-1), "resume_frame_idx" );
		
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
	
	const int width = vm["width"].as<int>();
	const int height = vm["height"].as<int>();
	std::cout << "resolution: " << width << ", " << height << std::endl;
	
	const std::string vx_filename_template = vm["vx_filename_template"].as<std::string>();
	std::cout << "vx_filename_template: " << vx_filename_template << std::endl;
	const std::string vy_filename_template = vm["vy_filename_template"].as<std::string>();
	std::cout << "vy_filename_template: " << vy_filename_template << std::endl;
	
	const int frame_start = vm["frame_start"].as<int>();
	const int frame_end = vm["frame_end"].as<int>();
	const int frame_skip = vm["frame_skip"].as<int>();
	std::cout << "frame start: " << frame_start << ", end: " << frame_end << ", skip: " << frame_skip << std::endl;
	
	const float angular_random_offset_deg = vm["angular_random_offset_deg"].as<float>();
	std::cout << "angular_random_offset_deg: " << angular_random_offset_deg << std::endl;
	const float random_offset_factor = vm["random_offset_factor"].as<float>();
	const float length_factor = vm["length_factor"].as<float>();
	const float width_factor = vm["width_factor"].as<float>();
	std::cout << "random_offset_factor: " << random_offset_factor << ", length_factor: " << length_factor << ", width_factor: " << width_factor << std::endl;
	const float length_random_factor_relative = vm["length_random_factor_relative"].as<float>();
	const float width_random_factor_relative = vm["width_random_factor_relative"].as<float>();
	std::cout << "length_random_factor_relative: " << length_random_factor_relative << ", width_random_factor_relative: " << width_random_factor_relative << std::endl;
	
	const float stroke_step_length = vm["stroke_step_length"].as<float>();
	const float stroke_step_length_accuracy = vm["stroke_step_length_accuracy"].as<float>();
	std::cout << "stroke_step_length: " << stroke_step_length << ", stroke_step_length_accuracy: " << 100.0 * stroke_step_length_accuracy << "%%" << std::endl;
	
	const float consecutive_failure_max = vm["consecutive_failure_max"].as<int>();
	std::cout << "consecutive_failure_max: " << consecutive_failure_max << std::endl;
	
	const float texture_start_max_random_offset = vm["texture_start_max_random_offset"].as<float>();
	const float texture_end_max_random_offset = vm["texture_end_max_random_offset"].as<float>();
	std::cout << "texture_start_max_random_offset: " << texture_start_max_random_offset << ", texture_end_max_random_offset: " << texture_end_max_random_offset << std::endl;
	
	const std::string sort_type = vm["sort_type"].as<std::string>();
	std::cout << "sort_type: " << sort_type << std::endl;
	
	const int _clip_with_undercoat_alpha = vm["clip_with_undercoat_alpha"].as<int>();
	const bool clip_with_undercoat_alpha = _clip_with_undercoat_alpha != 0;
	if( clip_with_undercoat_alpha )
		std::cout << "clip_with_undercoat_alpha: True" << std::endl;
	else
		std::cout << "clip_with_undercoat_alpha: False" << std::endl;
	
	const std::string texture_filename = vm["texture_filename"].as<std::string>();
	std::cout << "texture_filename: " << texture_filename << std::endl;
	const std::string texture_for_active_set_for_new_stroke_filename = vm["texture_for_active_set_for_new_stroke_filename"].as<std::string>();
	std::cout << "texture_for_active_set_for_new_stroke_filename: " << texture_for_active_set_for_new_stroke_filename << std::endl;
	const std::string texture_for_active_set_for_existing_stroke_filename = vm["texture_for_active_set_for_existing_stroke_filename"].as<std::string>();
	std::cout << "texture_for_active_set_for_existing_stroke_filename: " << texture_for_active_set_for_existing_stroke_filename << std::endl;
	
	const int num_textures = vm["num_textures"].as<int>();
	const int texture_length_mipmap_level = vm["texture_length_mipmap_level"].as<int>();
	std::cout << "num_textures: " << num_textures << ", texture_length_mipmap_level: " << texture_length_mipmap_level << std::endl;
	
	const std::string out_anchor_filename_template = vm["out_anchor_filename_template"].as<std::string>();
	std::cout << "out_anchor_filename_template: " << out_anchor_filename_template << std::endl;
	const std::string out_stroke_filename_template = vm["out_stroke_filename_template"].as<std::string>();
	std::cout << "out_stroke_filename_template: " << out_stroke_filename_template << std::endl;
	const std::string out_stroke_data_filename_template = vm["out_stroke_data_filename_template"].as<std::string>();
	std::cout << "out_stroke_data_filename_template: " << out_stroke_data_filename_template << std::endl;
	
	const std::string orientation_filename_template = vm["orientation_filename_template"].as<std::string>();
	std::cout << "orientation_filename_template: " << orientation_filename_template << std::endl;
	const std::string undercoat_filename_template = vm["undercoat_filename_template"].as<std::string>();
	std::cout << "undercoat_filename_template: " << undercoat_filename_template << std::endl;
	const std::string color_filename_template = vm["color_filename_template"].as<std::string>();
	std::cout << "color_filename_template: " << color_filename_template << std::endl;
	const std::string width_filename_template = vm["width_filename_template"].as<std::string>();
	std::cout << "width_filename_template: " << width_filename_template << std::endl;
	const std::string length_filename_template = vm["length_filename_template"].as<std::string>();
	std::cout << "length_filename_template: " << length_filename_template << std::endl;
	
	std::string mask_file_template = "";
	if( vm.count( "mask_file_template" ) ) 
		mask_file_template = vm["mask_file_template"].as<std::string>();
	std::cout << "mask_file_template: " << mask_file_template << std::endl;
	
	std::string active_set_file_template = "";
	if( vm.count( "active_set_file_template" ) ) 
		active_set_file_template = vm["active_set_file_template"].as<std::string>();
	std::cout << "active_set_file_template: " << active_set_file_template << std::endl;
	const float frame_rate = vm["frame_rate"].as<float>();
	std::cout << "frame_rate: " << frame_rate << std::endl;
	
	std::string region_label_template = "";
	if( vm.count( "region_label_template" ) )
		region_label_template = vm["region_label_template"].as<std::string>();
	std::cout << "region_label_template: " << region_label_template << std::endl;
	
	const bool remove_hidden_strokes = vm["remove_hidden_strokes"].as<int>() == 1 ? true : false;
	std::cout << "remove_hidden_strokes: " << remove_hidden_strokes << std::endl;
	const float remove_hidden_strokes_thr_contribution = vm["remove_hidden_strokes_thr_contribution"].as<float>();
	std::cout << "remove_hidden_strokes_thr_contribution: " << remove_hidden_strokes_thr_contribution << std::endl;
	const float remove_hidden_strokes_thr_alpha = vm["remove_hidden_strokes_thr_alpha"].as<float>();
	std::cout << "remove_hidden_strokes_thr_alpha: " << remove_hidden_strokes_thr_alpha << std::endl;
	
	std::chrono::steady_clock::time_point _begin, _end;
	
	std::random_device rd;
	std::mt19937 mt(1);
	std::uniform_real_distribution<> dist(0.0, 1.0);
	
	std::vector<float> max_random; 
	max_random.push_back( angular_random_offset_deg ); max_random.push_back( length_random_factor_relative );
	max_random.push_back( width_random_factor_relative ); max_random.push_back( 1.0 );
	max_random.push_back( texture_start_max_random_offset ); max_random.push_back( texture_end_max_random_offset );
	std::vector<bool> random_pm;
	random_pm.push_back( true ); random_pm.push_back( true ); random_pm.push_back( true );
	random_pm.push_back( false ); random_pm.push_back( false ); random_pm.push_back( false );
	
	StrokeRenderer* renderer = new StrokeRenderer( width, height );
	if( !renderer->initializeGL() )
		return -1;
	
	FrameBufferObject* fbo = new FrameBufferObject( width, height );
	std::cout << "FBO colorBuffer: " << fbo->colorBuffer() << std::endl;
	
	Image<float, 4> image;
	image.init( width, height );
	
	GLuint query = 0;
	glGenQueries( 1, &query );
	
	renderer->compileVisibilityTestShader( remove_hidden_strokes_thr_contribution, remove_hidden_strokes_thr_alpha );
	
	std::vector<int> remove_flag;
	
	Image<float, 4> texture;
	loadImagePng( texture_filename, texture );
	Image<float, 4> texture_for_new;
	loadImagePng( texture_for_active_set_for_new_stroke_filename, texture_for_new );
	Image<float, 4> texture_for_existing;
	loadImagePng( texture_for_active_set_for_existing_stroke_filename, texture_for_existing );
	
	if( texture.getWidth() != texture_for_new.getWidth() || texture.getHeight() != texture_for_new.getHeight() 
		|| texture.getWidth() != texture_for_existing.getWidth() || texture.getHeight() != texture_for_existing.getHeight() )	
	{
		std::cout << "ERROR!!!: mismatch in image size between texture and texture_for_updating_active_set..." << std::endl;
		exit(-1);
	}
	
	int tex_single_w, tex_single_h;
	float min_tex_ratio;
	
	TexAuxData tex_aux_data;
	compute_tex_settings( Eigen::Vector2i{ texture.getWidth(), texture.getHeight() }, num_textures, texture_length_mipmap_level, tex_aux_data );

  std::cout << "tex_single_w: " << tex_aux_data.single_w << std::endl;
  std::cout << "tex_single_h: " << tex_aux_data.single_h << std::endl;
	
	GLuint texture_index = renderer->setupStrokeTexture( texture );
	GLuint texture_new_index = renderer->setupStrokeTexture( texture_for_new );
	GLuint texture_existing_index = renderer->setupStrokeTexture( texture_for_existing );
	
	constexpr int buf_size = 4096;
	char buf[buf_size];
	
	unsigned char* image_bytes = (unsigned char*)malloc( sizeof(unsigned char) * width * height * 4 );
	HierarchicalActiveSet has( width, height );
	
	int next_id = 0;
	
	AnchorData anchors;
	StrokeData strokes;
	
	if( vm[ "resume_frame_idx" ].as<int>() >= 0 )
	{
		snprintf( buf, buf_size, out_anchor_filename_template.c_str(), vm[ "resume_frame_idx" ].as<int>() ); std::filesystem::path resume_anchor_path = buf;
		std::string resume_anchor_filename = resume_anchor_path.string();
		
		snprintf( buf, buf_size, out_stroke_data_filename_template.c_str(), vm[ "resume_frame_idx" ].as<int>() ); std::filesystem::path resume_stroke_data_path = buf;
		std::string resume_stroke_data_filename = resume_stroke_data_path.string();
		
		loadAnchors( resume_anchor_filename, next_id, anchors );
		int _width, _height;
		std::string _texture_filename;
		TexAuxData _aux;
		loadStrokes( resume_stroke_data_filename, _width, _height, strokes, _texture_filename, _aux );
		
		std::cout << "Resumed from frame " << vm[ "resume_frame_idx" ].as<int>() << std::endl;
		
		std::cout << "anchor size: " << anchors.pos.size() << std::endl;
		std::cout << "stroke size: " << strokes.strokes.size() << std::endl;
		std::cout << "next_id: " << next_id << std::endl;
	}
	
	std::string log_file_name = vm["log_file_name"].as<std::string>();

	std::ofstream log_ofs;
	log_ofs.open( log_file_name );
	log_ofs << "frame_idx, num_strokes, elapsed_time" << std::endl;
	log_ofs.close();
	
	for( int frame_idx = frame_start; frame_idx <= frame_end; frame_idx += frame_skip )
	{
		std::cout << "processing frame " << frame_idx << std::endl;
		_begin = std::chrono::steady_clock::now();
		
		// ### set up files
		snprintf( buf, buf_size, color_filename_template.c_str(), frame_idx ); 
		std::string color_filename = buf;

		snprintf( buf, buf_size, width_filename_template.c_str(), frame_idx ); 
		std::string width_filename = buf;

		snprintf( buf, buf_size, length_filename_template.c_str(), frame_idx );
		std::string length_filename = buf;

		snprintf( buf, buf_size, orientation_filename_template.c_str(), frame_idx );
		std::string orientation_filename = buf;

		snprintf( buf, buf_size, undercoat_filename_template.c_str(), frame_idx ); 
		std::string undercoat_filename = buf;
		
		snprintf( buf, buf_size, vx_filename_template.c_str(), frame_idx );
		std::string vx_path_str = buf;

		snprintf( buf, buf_size, vy_filename_template.c_str(), frame_idx );
		std::string vy_path_str = buf;
		
		std::string mask_filename = "";
		if( mask_file_template != "" )
		{
			snprintf( buf, buf_size, mask_file_template.c_str(), frame_idx ); 
			mask_filename = buf;
		}
	
		std::string active_set_filename = "";
		if( active_set_file_template != "" )
		{
			snprintf( buf, buf_size, active_set_file_template.c_str(), frame_idx ); 
			active_set_filename = buf;
		}
		
		std::string region_label_filename = "";
		if( region_label_template != "" )
		{
			snprintf( buf, buf_size, region_label_template.c_str(), frame_idx ); 
			region_label_filename = buf;
		}
		
		snprintf( buf, buf_size, out_anchor_filename_template.c_str(), frame_idx ); std::filesystem::path out_anchor_path = buf;
		std::string out_anchor_filename = out_anchor_path.string();


		if( !std::filesystem::is_directory( out_anchor_path.parent_path() ) ) std::filesystem::create_directory( out_anchor_filename );
	
		snprintf( buf, buf_size, out_stroke_filename_template.c_str(), frame_idx ); std::filesystem::path out_stroke_path = buf;
		std::string out_stroke_filename = out_stroke_path.string();

		if( !std::filesystem::is_directory( out_stroke_path.parent_path() ) ) std::filesystem::create_directory( out_stroke_filename );
		
		std::string out_stroke_data_filename = "";
		if( out_stroke_data_filename_template != "" )
		{
			snprintf( buf, buf_size, out_stroke_data_filename_template.c_str(), frame_idx ); std::filesystem::path out_stroke_data_path = buf;
			out_stroke_data_filename = out_stroke_data_path.string();
			if( !std::filesystem::is_directory( out_stroke_data_path.parent_path() ) ) std::filesystem::create_directory( out_stroke_data_filename );
		}
		
		OrientationSampler orientationSampler( orientation_filename );
		NearestSampler<3> colorSampler( color_filename );
		NearestSampler<2> lengthSampler( length_filename );
		NearestSampler<2> widthSampler( width_filename );
		VelocitySampler velocitySampler( vx_path_str, vy_path_str );
		Sampler* maskSampler = nullptr;
		if( mask_filename != "" )
			maskSampler = new NearestSampler<2>( mask_filename );
		else
		{
			float data[1]; data[0] = 1.0;
			maskSampler = new ConstantSampler<1>( data );
		}
		
		Sampler* regionLabelSampler = nullptr;
		if( region_label_filename != "" )
			regionLabelSampler = new RegionLabelSampler( region_label_filename );
		else
		{
			float data[1]; data[0] = 1.0;
			regionLabelSampler = new ConstantSampler<1>( data );
		}
		
		glUseProgram( renderer->defaultShader() );
		
		// ### draw default active_set
		renderer->clear();
		
		if( active_set_filename != "" )
		{
			Image<float, 4> default_active_set;
			loadImagePng( active_set_filename, default_active_set );
			
			GLuint default_active_set_texture_index = renderer->setupStrokeTexture( default_active_set );
			
			TexAuxData bg_tex_aux_data;
			compute_tex_settings( Eigen::Vector2i{ default_active_set.getWidth(), default_active_set.getHeight() }, 1, 1, bg_tex_aux_data );
			
			renderer->setTexAuxData( renderer->defaultShader(), bg_tex_aux_data );
			renderer->useTexture( default_active_set_texture_index );
			renderer->drawTextureAsQuad( renderer->defaultShader() );
		}
		
		renderer->readBufferBytes( image_bytes );
		has.setActiveSetFromRGBABufferBytes( image_bytes, width, height, 0.5 );
		
		// ### process existing strokes
		renderer->setTexAuxData( renderer->defaultShader(), tex_aux_data );
		std::cout << "stroke size: " << strokes.strokes.size() << std::endl;
		renderer->useTexture( texture_existing_index );
		
		AnchorData __anchors;
		
		updateAnchorPointsAndStrokes( strokes, renderer, image_bytes, &has, mt, 
			anchors, __anchors, tex_aux_data,
			&colorSampler, &lengthSampler, &widthSampler, &orientationSampler, maskSampler, regionLabelSampler,
			frame_idx, width, height, random_offset_factor, length_factor, width_factor, stroke_step_length, stroke_step_length_accuracy );
		
		anchors.setFromOther( __anchors );
		
		// ### process new strokes
		AnchorData _anchors;
		StrokeData _strokes;
		
		renderer->useTexture( texture_new_index );
		
		int _next_id = generateAnchorPointsAndStrokes( _anchors, _strokes, renderer, image_bytes, &has, mt, 
			max_random, random_pm, tex_aux_data, 
			&colorSampler, &lengthSampler, &widthSampler, &orientationSampler, maskSampler, regionLabelSampler,
			frame_idx, width, height, random_offset_factor, length_factor, width_factor, next_id, stroke_step_length, stroke_step_length_accuracy, consecutive_failure_max );
			
		std::cout << "added " << _next_id - next_id << " strokes" << std::endl;
		next_id = _next_id;
					
		// ### sorting		
		if( sort_type == "all_sort_luminance" )
			all_sort_strokes_frame_set_by_luminance( anchors, strokes, _anchors, _strokes );
		else if( sort_type == "add_sort_luminance" )
			sort_strokes_frame_set_by_luminance( anchors, strokes, _anchors, _strokes );
		else if( sort_type == "all_sort_index" )
			all_sort_strokes_frame_set_by_index( anchors, strokes, _anchors, _strokes );
		else // "add_sort_index"
			sort_strokes_frame_set_by_index( anchors, strokes, _anchors, _strokes );
				
		// ### remove hidden strokes
		if( remove_hidden_strokes )
		{
			fbo->bind();
			renderer->clearB();
			fbo->unbind();
			renderer->clear();
			remove_flag.resize( std::max<int>( 256, strokes.strokes.size() ) );
		
			renderer->useTexture( texture_index );
			glUseProgram( renderer->defaultShader() );
			renderer->setTexAuxData( renderer->defaultShader(), tex_aux_data );
			glUseProgram( renderer->visibilityTestShader() );
			renderer->setTexAuxData( renderer->visibilityTestShader(), tex_aux_data );
			
			for( int i=strokes.strokes.size()-1; i>=0; i-- )
			{	
				glBeginQuery( GL_ANY_SAMPLES_PASSED, query );
				glDisable( GL_BLEND );
				glUseProgram( renderer->visibilityTestShader() );
				renderer->useTexture( texture_index );
				glActiveTexture( GL_TEXTURE1 );
				glBindTexture( GL_TEXTURE_2D, fbo->colorBuffer() );
				renderer->draw( renderer->visibilityTestShader(), strokes.strokes[i], strokes.tex_info_int[i], strokes.tex_info_float[i] );
				glEndQuery( GL_ANY_SAMPLES_PASSED );
				GLuint ret = 2;
				glGetQueryObjectuiv( query, GL_QUERY_RESULT, &ret );
				
				remove_flag[i] = ( ret == GL_FALSE ) ? 1 : 0;
			
				fbo->bind();
		
				glEnable( GL_BLEND );
	    	glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
	    	glBlendFuncSeparate( GL_ZERO, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA );
		
				glUseProgram( renderer->defaultShader() );
				renderer->useTexture( texture_new_index );
				renderer->draw( renderer->defaultShader(), strokes.strokes[i], strokes.tex_info_int[i], strokes.tex_info_float[i] );
			
				fbo->unbind();
			}
					
			int count = 0;
			for( int i=0; i<strokes.strokes.size(); i++ )
			{
				if( remove_flag[i] == 1 )
					count++;
			}
			std::cout << "#strokes to remove: " << count << std::endl;
			anchors.remove( remove_flag );
			strokes.remove( remove_flag );
		
    	glEnable( GL_BLEND );
    	glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    	glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		}
		
		std::cout << "hidden strokes removed" << std::endl;
		
		// ### export
		if( out_anchor_filename != "" )
		{
			saveAnchors( out_anchor_filename, next_id, anchors );
		}
		
		if( out_stroke_data_filename != "" )
		{
			std::filesystem::path tex_path = "../"; tex_path /= texture_filename;
			std::string tex_fn = tex_path.string();
			saveStrokes( out_stroke_data_filename, width, height, strokes, tex_fn, tex_aux_data );
		}
		
		std::cout << "data saved" << std::endl;
		
		renderer->useTexture( texture_index );
		renderer->clear();
		renderer->draw( renderer->defaultShader(), strokes );
		
		renderer->readBuffer( image );
		saveImagePng( out_stroke_filename, image );
		
		std::cout << "image saved" << std::endl;
			
		// ### advection 
		advectPointsTVDRK3( anchors.pos, 1.0 / frame_rate, 5, &velocitySampler );
		
		std::cout << "anchor points advected" << std::endl;
		
		if( maskSampler != nullptr )
			delete maskSampler;
		
		_end = std::chrono::steady_clock::now();
		const float elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _begin).count() / 1000000.0;
		std::cout << "Elapsed time: " << elapsed_time << " [s]" << std::endl;
		
		log_ofs.open( log_file_name, std::ios::app );
		log_ofs << frame_idx << ", " << strokes.strokes.size() << ", " << elapsed_time << std::endl;
		log_ofs.close();
	}
	
	glDeleteQueries( 1, &query );
	free( image_bytes );
	renderer->finalize();
	
	return 0;
}
