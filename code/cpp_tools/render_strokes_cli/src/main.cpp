// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: render_strokes_cli/src/main.cpp
// Maintainer: Yonghao Yue and Hideki Todo
//
// Description:
// This file implements the final rendering step for strokes in an oil painting 
// style (to account for lighting and bump mapping kind effect).
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

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Core>

#include <iostream>
#include <filesystem>
#include <set>

#include <lib/image/image.h>
#include <lib/image/imageioutil.h>
#include <lib/image/imageiopng.h>
#include <HDF5File.h>

#include <random>
#include <chrono>
#include <cmath>

#include <boost/program_options.hpp>

#include <lib/strokes/framebufferobject.h>
#include <lib/strokes/stroke.h>

struct LightingParams
{
	float combine_height_top_factor;
	float combine_height_additive_factor;
	float combine_height_additive_log_factor;
	float tex_step;
	float height_scale;
	float vz;
	float lx;
	float ly;
	float lz;
	float glossiness;
	float kd;
	float ks;
	float ka;
	float light_intensity;
	float canvas_scale;
};

class StrokeRenderer
{
	StrokeRenderer();
public:
	StrokeRenderer( const int in_width, const int in_height, const LightingParams& in_params )
		: m_Width( in_width ), m_Height( in_height ), m_Window( nullptr ), m_BufferBytes( nullptr ), m_LightingParams( in_params )
	{
		m_Buffer.init( in_width, in_height );
		m_BufferBytes = (unsigned char*)malloc( sizeof(unsigned char) * in_width * in_height * 4 );
	}
		
	~StrokeRenderer()
	{
		free( m_BufferBytes );
		m_BufferBytes = nullptr;
		
		for( auto p: m_GeneratedTextures )
		{
			GLuint _tid = p;
			glDeleteTextures( 1, &_tid );
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
		
    compileColorShader();
		compileHeightShader();
		compileCombineHeightShader();
		compileOilPaintingShader();
			
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
		return true;
	}
		
	void compileColorShader()
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
		"uniform sampler2D colorTex;\n"
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
		"  finalColor = strokeColor * texture(colorTex, warped_uv);\n"
		"}";
  
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_ColorShader = glCreateProgram();
	  glAttachShader( m_ColorShader, fs );
	  glAttachShader( m_ColorShader, vs );
	  glLinkProgram( m_ColorShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_ColorShader );
    const GLuint MatrixID = glGetUniformLocation( m_ColorShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint colorTexLoc = glGetUniformLocation( m_ColorShader, "colorTex" );
		glUniform1i( colorTexLoc, 0 );
	}
	
	void compileHeightShader()
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
		"uniform sampler2D heightTex;\n"
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
		"  float h = texture(heightTex, warped_uv).x;\n"
		"  float alpha = texture(heightTex, warped_uv).w;\n"
		"  if( alpha <= 1.0/256.0 )\n"
		"    discard;\n"
		"  else\n"
		"    finalColor = vec4( 1.0/256.0, 1.0/256.0, 1.0/256.0, h );\n"
		"}";
  
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_HeightShader = glCreateProgram();
	  glAttachShader( m_HeightShader, fs );
	  glAttachShader( m_HeightShader, vs );
	  glLinkProgram( m_HeightShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_HeightShader );
    const GLuint MatrixID = glGetUniformLocation( m_HeightShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint heightTexLoc = glGetUniformLocation( m_HeightShader, "heightTex" );
		glUniform1i( heightTexLoc, 0 );
	}
	
	void compileCombineHeightShader()
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
		"uniform sampler2D heightBuf;\n"
		"uniform float top_factor;\n"
		"uniform float additive_factor;\n"
		"uniform float additive_log_factor;\n"
		"out vec4 finalColor;\n"
	  "void main() {\n"
		"  float additive = texture(heightBuf, fragmentUV).x;\n"
		"  float top = texture(heightBuf, fragmentUV).w;\n"
		"  float h = top * top_factor + log( 1.0 + additive_log_factor * additive * 256.0 ) * additive_factor;\n"
		"  float mask = additive > 0.0 ? 1.0 : 0.0;\n"
		"  finalColor = vec4( h, h, h, mask );\n"
		"}";
	
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_CombineHeightShader = glCreateProgram();
	  glAttachShader( m_CombineHeightShader, fs );
	  glAttachShader( m_CombineHeightShader, vs );
	  glLinkProgram( m_CombineHeightShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_CombineHeightShader );
    const GLuint MatrixID = glGetUniformLocation( m_CombineHeightShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint heightBufLoc = glGetUniformLocation( m_CombineHeightShader, "heightBuf" );
		glUniform1i( heightBufLoc, 0 );
		const GLuint topFactorLoc = glGetUniformLocation( m_CombineHeightShader, "top_factor" );
		glUniform1f( topFactorLoc, m_LightingParams.combine_height_top_factor );
		const GLuint additiveFactorLoc = glGetUniformLocation( m_CombineHeightShader, "additive_factor" );
		glUniform1f( additiveFactorLoc, m_LightingParams.combine_height_additive_factor );
		const GLuint additiveLogFactorLoc = glGetUniformLocation( m_CombineHeightShader, "additive_log_factor" );
		glUniform1f( additiveLogFactorLoc, m_LightingParams.combine_height_additive_log_factor );
	}
	
	void compileOilPaintingShader()
	{
	  const char* vertex_shader =
	  "#version 400\n"
	  "in vec2 vp;\n"
		"layout(location = 1) in vec2 uv;\n"
	  "uniform mat4 MVP;\n"
		"out vec2 fragmentUV;\n"
		"out vec3 pos;\n"
	  "void main() {\n"
	  "  gl_Position = MVP * vec4(vp, 0.0, 1.0);\n"
		"  fragmentUV = uv;\n"
		"  pos = vec3( gl_Position.x, gl_Position.y, 0.0 );\n"
	  "}";
  
	  const char* fragment_shader =
	  "#version 400\n"
		"in vec2 fragmentUV;\n"
		"in vec3 pos;\n"
		"uniform sampler2D colorTex;\n"
		"uniform sampler2D heightTex;\n"
		"uniform vec2 texStep;\n"
		"uniform float heightScale;\n"
		"uniform float glossiness;\n"
		"uniform float lightIntensity;\n"
		"uniform float canvasScale;\n"
		"uniform float kd;\n"
		"uniform float ks;\n"
		"uniform float ka;\n"
		"uniform vec3 viewPoint;\n"
		"uniform vec3 lightPoint;\n"
	  "out vec4 finalColor;\n"
	  "void main() {\n"
		"  float h_c = heightScale * texture(heightTex, fragmentUV).x;\n"
		"  float h_x = heightScale * texture(heightTex, vec2(fragmentUV.x+texStep.x, fragmentUV.y) ).x;\n"
		"  float h_y = heightScale * texture(heightTex, vec2(fragmentUV.x, fragmentUV.y-texStep.y) ).x;\n"
		"  float mask = texture(heightTex, fragmentUV).w;\n"
		"  vec3 _n = vec3( -( h_x - h_c ) / texStep.x, -( h_y - h_c ) / texStep.y, 1.0 );\n"
		"  vec3 n = normalize( _n );\n"
		"  vec3 dv = normalize( viewPoint - pos*canvasScale );\n"
		"  vec3 dl = normalize( lightPoint - pos*canvasScale );\n"
		"  vec3 dh = normalize( ( dv + dl ) * 0.5 );\n"
		"  float _dot = dot( n, dh );\n"
		"  float _dot_n_dl = dot( n, dl );\n"
		"  float nbp = ( glossiness + 2.0 ) * ( glossiness + 0.4 ) / ( 8.0 * 3.1415926538 * ( exp2( - glossiness * 0.5 ) + glossiness ) );\n"
		"  float pwr_dot = _dot_n_dl * nbp * pow( _dot, glossiness );\n"
		"  vec4 col = texture(colorTex, fragmentUV);\n"
		"  vec4 diffuseColor = lightIntensity * mask * kd * col * vec4( _dot_n_dl, _dot_n_dl, _dot_n_dl, 1.0 );\n"
		"  vec4 specularColor = lightIntensity * mask * ks * vec4( pwr_dot, pwr_dot, pwr_dot, 1.0 );\n"
		"  vec4 ambientColor = ka * col;\n"
		"  finalColor = ambientColor + diffuseColor + specularColor;\n"
		"}";
  
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_OilPaintingShader = glCreateProgram();
	  glAttachShader( m_OilPaintingShader, fs );
	  glAttachShader( m_OilPaintingShader, vs );
	  glLinkProgram( m_OilPaintingShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_OilPaintingShader );
    const GLuint MatrixID = glGetUniformLocation( m_OilPaintingShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint colorTexLoc = glGetUniformLocation( m_OilPaintingShader, "colorTex" );
		glUniform1i( colorTexLoc, 0 );
		const GLuint heightTexLoc = glGetUniformLocation( m_OilPaintingShader, "heightTex" );
		glUniform1i( heightTexLoc, 1 );
		const GLuint texStepLoc = glGetUniformLocation( m_OilPaintingShader, "texStep" );
		glUniform2f( texStepLoc, m_LightingParams.tex_step, m_LightingParams.tex_step );
		const GLuint heightScaleLoc = glGetUniformLocation( m_OilPaintingShader, "heightScale" );
		glUniform1f( heightScaleLoc, m_LightingParams.height_scale );
		const GLuint viewPointLoc = glGetUniformLocation( m_OilPaintingShader, "viewPoint" );
		glUniform3f( viewPointLoc, 0.0, 0.0, m_LightingParams.vz );
		const GLuint lightPointLoc = glGetUniformLocation( m_OilPaintingShader, "lightPoint" );
		glUniform3f( lightPointLoc, m_LightingParams.lx, m_LightingParams.ly, m_LightingParams.lz );
		const GLuint glossinessLoc = glGetUniformLocation( m_OilPaintingShader, "glossiness" );
		glUniform1f( glossinessLoc, m_LightingParams.glossiness );
		const GLuint kdLoc = glGetUniformLocation( m_OilPaintingShader, "kd" );
		glUniform1f( kdLoc, m_LightingParams.kd );
		const GLuint ksLoc = glGetUniformLocation( m_OilPaintingShader, "ks" );
		glUniform1f( ksLoc, m_LightingParams.ks );
		const GLuint kaLoc = glGetUniformLocation( m_OilPaintingShader, "ka" );
		glUniform1f( kaLoc, m_LightingParams.ka );
		const GLuint lightIntensityLoc = glGetUniformLocation( m_OilPaintingShader, "lightIntensity" );
		glUniform1f( lightIntensityLoc, m_LightingParams.light_intensity );
		const GLuint canvasScaleLoc = glGetUniformLocation( m_OilPaintingShader, "canvasScale" );
		glUniform1f( canvasScaleLoc, m_LightingParams.canvas_scale );
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
		
		m_GeneratedTextures.insert( texture );
		
		return texture;
	}
	
	void useTexture( GLuint in_Texture )
	{
		m_StrokeTexture = in_Texture;	
	}
	
	void deleteTexture( GLuint in_Texture )
	{
		auto q = m_GeneratedTextures.find( in_Texture );
		if( q != m_GeneratedTextures.end() )
		{
			GLuint _tid = in_Texture;
			glDeleteTextures( 1, &_tid );
			m_GeneratedTextures.erase( q );
		}
		else
		{
			std::cout << "GL Texture with id " << in_Texture << " was not created by the renderer. Please delete this texture manually. This might be a bug." << std::endl;
		}
	}
	
	void clear()
	{
    glViewport( 0, 0, m_Width, m_Height );
		glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	}
	
	void setTexAuxData( GLuint in_Shader, const TexAuxData& in_aux )
	{
		glUseProgram( in_Shader );
		const GLuint singleTexSizeLoc = glGetUniformLocation( in_Shader, "singleTexSize" );
		float singleTexSizeVal[2] = { float( in_aux.single_w ), float( in_aux.single_h ) };
		glUniform2f( singleTexSizeLoc, singleTexSizeVal[0], singleTexSizeVal[1] );
		const GLuint texSizeLoc = glGetUniformLocation( in_Shader, "texSize" );
		float texSizeVal[2] = { float( in_aux.size(0) ), float( in_aux.size(1) ) };
		glUniform2f( texSizeLoc, texSizeVal[0], texSizeVal[1] );
	}
		
	void _draw( GLuint in_Shader, const Stroke& in_stroke, const Eigen::Vector2i& in_info_int, const Eigen::Vector2f& in_info_float )
	{
		float color[4]; in_stroke.color( color );
		const GLuint colorLoc = glGetUniformLocation( in_Shader, "strokeColor" );
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
	
	void _draw( GLuint in_Shader, const StrokeData& in_strokes )
	{
		for( int i=0; i<in_strokes.strokes.size(); i++ )
		{
			_draw( in_Shader, in_strokes.strokes[i], in_strokes.tex_info_int[i], in_strokes.tex_info_float[i] );
		}
	}
	
	void drawHeight( const StrokeData& in_strokes )
	{
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_ONE, GL_ONE, GL_ONE, GL_ZERO );
		
		glUseProgram( m_HeightShader );
		_draw( m_HeightShader, in_strokes );
	}
	
	void drawColor( const StrokeData& in_strokes )
	{
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
		glUseProgram( m_ColorShader );
		_draw( m_ColorShader, in_strokes );
	}
	
	void drawTextureFull()
	{	
		const int max_res = std::max<int>( m_Width, m_Height );
		const float w_max = 2.0 * m_Width / max_res - 1.0;
		const float h_max = 2.0 * m_Height / max_res - 1.0;
		
		float points[] = {
			-1.0f, h_max,
			-1.0f, -1.0f,
			w_max,  h_max,
			-1.0f, -1.0f,
			w_max, -1.0f,
			w_max, h_max
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
  	
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * 2 );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteVertexArrays( 1, &vao );
	}
	
	void drawTextureFullFlip()
	{	
		const int max_res = std::max<int>( m_Width, m_Height );
		const float w_max = 2.0 * m_Width / max_res - 1.0;
		const float h_max = 2.0 * m_Height / max_res - 1.0;
		
		float points[] = {
			-1.0f, -1.0f,
			-1.0f, h_max,
			w_max,  -1.0f,
			-1.0f, h_max,
			w_max, h_max,
			w_max, -1.0f
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
  	
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * 2 );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteVertexArrays( 1, &vao );
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
			//*
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
			//*
			unsigned char* p_src = &( m_BufferBytes[ ( j * in_w ) * 4 ] );
			unsigned char* p_dst = &( io_buffer[ ( (j+in_y)*in_buffer_width+(in_x) ) * 4 ] );
			
			for( int i=0; i<in_w*4; i++ )
			{
				*p_dst++=*p_src++;
			}
		}
	}
	
	GLuint colorShader()
	{
		return m_ColorShader;
	}
	
	GLuint heightShader()
	{
		return m_HeightShader;
	}
	
	GLuint combineHeightShader()
	{
		return m_CombineHeightShader;
	}
	
	GLuint oilPaintingShader()
	{
		return m_OilPaintingShader;
	}
		
private:
	int m_Width;
	int m_Height;
	GLFWwindow* m_Window;
	//GLuint m_Shader;
	GLuint m_HeightShader;
	GLuint m_CombineHeightShader;
	GLuint m_ColorShader;
	GLuint m_OilPaintingShader;
	GLuint m_StrokeTexture;
	glm::mat4 m_MVP;
	
	Image<float, 4> m_Buffer;
	unsigned char* m_BufferBytes;
	
	std::set<GLuint> m_GeneratedTextures;
	
	LightingParams m_LightingParams;
};

int main( int argc, char* argv[] )
{
	boost::program_options::options_description opt( "Option" );
	opt.add_options()
		("help", "help")
		("width,w", boost::program_options::value<int>()->default_value(1024), "width")
		("height,h", boost::program_options::value<int>()->default_value(1024), "height")
		("stroke_dir", boost::program_options::value<std::string>()->default_value("stroke"), "stroke_dir" )
		("frame_start", boost::program_options::value<int>()->default_value(140), "frame_start" )
		("frame_end", boost::program_options::value<int>()->default_value(141), "frame_end" )
		("frame_skip", boost::program_options::value<int>()->default_value(1), "frame_skip" )
		("color_texture_filename", boost::program_options::value<std::string>()->default_value("../../textures/texture.png"), "color_texture_filename" )
		("height_texture_filename", boost::program_options::value<std::string>()->default_value("../../textures/height.png"), "height_texture_filename" )
		("num_textures", boost::program_options::value<int>()->default_value(1), "num_textures" )
		("texture_length_mipmap_level", boost::program_options::value<int>()->default_value(1), "texture_length_mipmap_level" )
		("stroke_data_filename_template", boost::program_options::value<std::string>()->default_value("stroke_data/stroke_%03d.h5"), "stroke_data_filename_template" )
		("out_final_filename_template", boost::program_options::value<std::string>()->default_value("final/final_%03d.png"), "out_final_filename_template" )
		("undercoat_filename_template", boost::program_options::value<std::string>()->default_value("color/color_%03d.png"), "undercoat_filename_template" )
		("mask_file_template", boost::program_options::value<std::string>()->default_value(""), "mask_file_template" )
	  ("combine_height_top_factor", boost::program_options::value<float>()->default_value( exp2( 0.0 ) ), "combine_height_top_factor")
		("combine_height_additive_factor", boost::program_options::value<float>()->default_value( exp2( -2.0 ) ), "combine_height_additive_factor")
		("combine_height_additive_log_factor", boost::program_options::value<float>()->default_value( exp2( 0.0 ) ), "combine_height_additive_log_factor")
		("tex_step", boost::program_options::value<float>()->default_value( exp2( -7.953 ) ), "tex_step")
		("height_scale", boost::program_options::value<float>()->default_value( exp2( -9.212 ) ), "height_scale")
		("vz", boost::program_options::value<float>()->default_value( 3.2 ), "vz")
		("lx", boost::program_options::value<float>()->default_value( 0.0 ), "lx")
		("ly", boost::program_options::value<float>()->default_value( 1.2 ), "ly")
		("lz", boost::program_options::value<float>()->default_value( 3.2 ), "lz")
		("glossiness", boost::program_options::value<float>()->default_value( exp2( 8.130 ) ), "glossiness")
		("kd", boost::program_options::value<float>()->default_value( 0.24 ), "kd")
		("ks", boost::program_options::value<float>()->default_value( 0.01 ), "ks")
		("ka", boost::program_options::value<float>()->default_value( 0.29 ), "ka")
		("light_intensity", boost::program_options::value<float>()->default_value( exp2( 1.53 ) ), "light_intensity")
		("canvas_scale", boost::program_options::value<float>()->default_value( 0.4 ), "canvas_scale")
		("log_file_name", boost::program_options::value<std::string>()->default_value("stroke/log.txt"), "log_file_name" )
		("subset_ratio", boost::program_options::value<float>()->default_value( -1.0 ), "subset_ratio" ); // useful to make a video showing the drawing order of strokes
	
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
	
	const std::string stroke_dir = vm["stroke_dir"].as<std::string>();
	std::cout << "stroke_dir: " << stroke_dir << std::endl;
	
	const int frame_start = vm["frame_start"].as<int>();
	const int frame_end = vm["frame_end"].as<int>();
	const int frame_skip = vm["frame_skip"].as<int>();
	std::cout << "frame start: " << frame_start << ", end: " << frame_end << ", skip: " << frame_skip << std::endl;
			
	const std::string color_texture_filename = vm["color_texture_filename"].as<std::string>();
	std::cout << "color_texture_filename: " << color_texture_filename << std::endl;
	const std::string height_texture_filename = vm["height_texture_filename"].as<std::string>();
	std::cout << "height_texture_filename: " << height_texture_filename << std::endl;
	
	const int num_textures = vm["num_textures"].as<int>();
	const int texture_length_mipmap_level = vm["texture_length_mipmap_level"].as<int>();
	std::cout << "num_textures: " << num_textures << ", texture_length_mipmap_level: " << texture_length_mipmap_level << std::endl;
	
	const std::string stroke_data_filename_template = vm["stroke_data_filename_template"].as<std::string>();
	std::cout << "stroke_data_filename_template: " << stroke_data_filename_template << std::endl;
	const std::string out_final_filename_template = vm["out_final_filename_template"].as<std::string>();
	std::cout << "out_final_filename_template: " << out_final_filename_template << std::endl;
	
	const std::string undercoat_filename_template = vm["undercoat_filename_template"].as<std::string>();
	std::cout << "undercoat_filename_template: " << undercoat_filename_template << std::endl;
	
	std::string mask_file_template = "";
	if( vm.count( "mask_file_template" ) ) 
		mask_file_template = vm["mask_file_template"].as<std::string>();
	std::cout << "mask_file_template: " << mask_file_template << std::endl;
	
	LightingParams lighting_params;
	lighting_params.combine_height_top_factor = vm["combine_height_top_factor"].as<float>();
	lighting_params.combine_height_additive_factor = vm["combine_height_additive_factor"].as<float>();
	lighting_params.combine_height_additive_log_factor = vm["combine_height_additive_log_factor"].as<float>();
	lighting_params.tex_step = vm["tex_step"].as<float>();
	lighting_params.height_scale = vm["height_scale"].as<float>();
	lighting_params.vz = vm["vz"].as<float>();
	
	lighting_params.lx = vm["lx"].as<float>();
	lighting_params.ly = vm["ly"].as<float>();
	lighting_params.lz = vm["lz"].as<float>();
	
	lighting_params.glossiness = vm["glossiness"].as<float>();
	
	lighting_params.kd = vm["kd"].as<float>();
	lighting_params.ks = vm["ks"].as<float>();
	lighting_params.ka = vm["ka"].as<float>();
	
	lighting_params.light_intensity = vm["light_intensity"].as<float>();
	lighting_params.canvas_scale = vm["canvas_scale"].as<float>();	
	
	std::chrono::steady_clock::time_point _begin, _end;
	
	StrokeRenderer* renderer = new StrokeRenderer( width, height, lighting_params );
	if( !renderer->initializeGL() )
		return -1;
	
	FrameBufferObject* fbo_color = new FrameBufferObject( width, height );
	FrameBufferObject* fbo_height0 = new FrameBufferObject( width, height );
	FrameBufferObject* fbo_height = new FrameBufferObject( width, height );
	
	Image<float, 4> image;
	image.init( width, height );
	
	Image<float, 4> color_texture;
	loadImagePng( color_texture_filename, color_texture );
	Image<float, 4> height_texture;
	loadImagePng( height_texture_filename, height_texture );
	
	if( color_texture.getWidth() != height_texture.getWidth() || color_texture.getHeight() != height_texture.getHeight() )	
	{
		std::cout << "ERROR!!!: mismatch in image size between color_texture and height_texture..." << std::endl;
		exit(-1);
	}
	
	int tex_single_w, tex_single_h;
	float min_tex_ratio;
	
	TexAuxData tex_aux_data;
	compute_tex_settings( Eigen::Vector2i{ color_texture.getWidth(), color_texture.getHeight() }, num_textures, texture_length_mipmap_level, tex_aux_data );

  std::cout << "tex_single_w: " << tex_aux_data.single_w << std::endl;
  std::cout << "tex_single_h: " << tex_aux_data.single_h << std::endl;
	
	GLuint color_texture_index = renderer->setupStrokeTexture( color_texture );
	GLuint height_texture_index = renderer->setupStrokeTexture( height_texture );
	
	constexpr int buf_size = 4096;
	char buf[buf_size];
	
	unsigned char* image_bytes = (unsigned char*)malloc( sizeof(unsigned char) * width * height * 4 );
	
	StrokeData strokes;
	
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
		snprintf( buf, buf_size, undercoat_filename_template.c_str(), frame_idx ); std::filesystem::path undercoat_path = stroke_dir; undercoat_path /= buf;
		std::string undercoat_filename = undercoat_path.string();
		
		std::string mask_filename = "";
		if( mask_file_template != "" )
		{
			snprintf( buf, buf_size, mask_file_template.c_str(), frame_idx ); std::filesystem::path mask_path = stroke_dir; mask_path /= buf;
			mask_filename = mask_path.string();
		}
		
		snprintf( buf, buf_size, stroke_data_filename_template.c_str(), frame_idx ); std::filesystem::path stroke_data_path = stroke_dir; stroke_data_path /= buf;
		std::string stroke_data_filename = stroke_data_path.string();

		snprintf( buf, buf_size, out_final_filename_template.c_str(), frame_idx ); std::filesystem::path out_final_path = stroke_dir; out_final_path /= buf;
		std::string out_final_filename = out_final_path.string();
		
		int _width, _height;
		std::string _texture_filename;
		TexAuxData _aux;
		std::cout << "stroke_data_filename: " << stroke_data_filename << std::endl;
		loadStrokes( stroke_data_filename, _width, _height, strokes, _texture_filename, _aux ); 
		if( color_texture.getWidth() != _aux.size(0) || color_texture.getHeight() != _aux.size(1) )	
			std::cout << "WARNING!!!: mismatch in image size between color_texture and texture associated with stroke data..." << std::endl;
				
		const float subset_ratio = vm[ "subset_ratio" ].as<float>();
		if( subset_ratio >= 0.0 && subset_ratio < 1.0 )
		{
			strokes.shrinkSize( subset_ratio );
		}
		
		fbo_color->bind();
		renderer->clear();
		renderer->useTexture( color_texture_index );
		renderer->setTexAuxData( renderer->colorShader(), tex_aux_data );
		renderer->drawColor( strokes );
		fbo_color->unbind();
				
		fbo_height0->bind();
		renderer->clear();
		renderer->useTexture( height_texture_index );
		renderer->setTexAuxData( renderer->heightShader(), tex_aux_data );
		renderer->drawHeight( strokes );
		fbo_height0->unbind();
				
		fbo_height->bind();
		renderer->clear();
		glUseProgram( renderer->combineHeightShader() );
		glDisable( GL_BLEND );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, fbo_height0->colorBuffer() );
		renderer->drawTextureFull();
		fbo_height->unbind();
		
		renderer->clear();
		glUseProgram( renderer->oilPaintingShader() );
		glDisable( GL_BLEND );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, fbo_color->colorBuffer() );
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture( GL_TEXTURE_2D, fbo_height->colorBuffer() );
		renderer->drawTextureFullFlip();
		
		renderer->readBuffer( image );
		saveImagePngInvertY( out_final_filename, image );
				
		_end = std::chrono::steady_clock::now();
		const float elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _begin).count() / 1000000.0;
		std::cout << "Elapsed time: " << elapsed_time << " [s]" << std::endl;
		
		log_ofs.open( log_file_name, std::ios::app );
		log_ofs << frame_idx << ", " << strokes.strokes.size() << ", " << elapsed_time << std::endl;
		log_ofs.close();
	}
	
	free( image_bytes );
	renderer->finalize();
	
	return 0;
}
