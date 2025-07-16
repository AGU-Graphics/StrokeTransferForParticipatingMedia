// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: oil_lighting_editor_gui/src/main.cpp
// Maintainer: Yonghao Yue
//
// Description:
// This file implements an user interface to adjust shader parameters for 
// the lighting model used in oil painting rendering (render_strokes_cli), 
// which emulates a simple bump mapping kind effect.
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

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuiFileDialog.h>

#include <HDF5File.h>

#include <Eigen/Core>

#include <iostream>
#include <set>

#include <lib/image/image.h>
#include <lib/image/imageioutil.h>
#include <lib/image/imageiopng.h>

#include <random>
#include <chrono>

#include <lib/strokes/framebufferobject.h>
#include <lib/strokes/stroke.h>

float g_FrameSize_WindowSize_Scale_x = 1.0;
float g_FrameSize_WindowSize_Scale_y = 1.0;

float g_CombineHeight_TopFactor = exp2( 0.0 ); 
float g_CombineHeight_AdditiveFactor = exp2( -2.0 ); 
float g_CombineHeight_AdditiveLogFactor = exp2( 0.0 ); 

float g_TexStepX = exp2( -7.953 ); 
float g_TexStepY = exp2( -7.953 ); 
float g_HeightScale = exp2( -9.212 ); 
float g_Vz = 3.2;
float g_Lx = 0.0; 
float g_Ly = 1.2; 
float g_Lz = 3.2;
float g_Glossiness = exp2( 8.130 ); 
float g_kd = 0.24; 
float g_ks = 0.01; 
float g_ka = 0.29; 
float g_LightIntensity = exp2( 1.53 ); 

float g_CanvasScale = 0.4;

void printParameters()
{
	std::cout << "##############################" << std::endl;
	std::cout << "g_CombineHeight_TopFactor: " << g_CombineHeight_TopFactor << std::endl;
	std::cout << "g_CombineHeight_AdditiveFactor: " << g_CombineHeight_AdditiveFactor << std::endl;
	std::cout << "g_CombineHeight_AdditiveLogFactor: " << g_CombineHeight_AdditiveLogFactor << std::endl;
	
	std::cout << "g_TexStepX: " << g_TexStepX << std::endl;
	std::cout << "g_TexStepY: " << g_TexStepY << std::endl;
	std::cout << "g_HeightScale: " << g_HeightScale << std::endl;
	
	std::cout << "g_Vz: " << g_Vz << std::endl;
	std::cout << "g_Lx: " << g_Lx << std::endl;
	std::cout << "g_Ly: " << g_Ly << std::endl;
	std::cout << "g_Lz: " << g_Lz << std::endl;
	
	std::cout << "g_Glossiness: " << g_Glossiness << std::endl;
	std::cout << "g_kd: " << g_kd << std::endl;
	std::cout << "g_ks: " << g_ks << std::endl;
	std::cout << "g_ka: " << g_ka << std::endl;
	std::cout << "g_LightIntensity: " << g_LightIntensity << std::endl;
	
	std::cout << "g_CanvasScale: " << g_CanvasScale << std::endl;
}

class Renderer
{
	Renderer();
public:
	Renderer( const int in_width, const int in_height )
		: m_Width( in_width ), m_Height( in_height ), m_BufferBytes( nullptr )
	{
		m_Buffer.init( in_width, in_height );
		m_BufferBytes = (unsigned char*)malloc( sizeof(unsigned char) * in_width * in_height * 4 );
	}
		
	~Renderer()
	{
		free( m_BufferBytes );
		m_BufferBytes = nullptr;
		
		for( auto p: m_Textures )
		{
			GLuint _tid = p;
			glDeleteTextures( 1, &_tid );
		}
		m_Textures.clear();
	}
	
	bool initializeGL()
	{		
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
    compileColorShader();
		compileHeightShader();
		compileCombineHeightShader();
		compileOilPaintingShader();
		
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
		glUniform1f( topFactorLoc, g_CombineHeight_TopFactor );
		const GLuint additiveFactorLoc = glGetUniformLocation( m_CombineHeightShader, "additive_factor" );
		glUniform1f( additiveFactorLoc, g_CombineHeight_AdditiveFactor );
		const GLuint additiveLogFactorLoc = glGetUniformLocation( m_CombineHeightShader, "additive_log_factor" );
		glUniform1f( additiveLogFactorLoc, g_CombineHeight_AdditiveLogFactor );
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
		glUniform2f( texStepLoc, g_TexStepX, g_TexStepY );
		const GLuint heightScaleLoc = glGetUniformLocation( m_OilPaintingShader, "heightScale" );
		glUniform1f( heightScaleLoc, g_HeightScale );
		const GLuint viewPointLoc = glGetUniformLocation( m_OilPaintingShader, "viewPoint" );
		glUniform3f( viewPointLoc, 0.0, 0.0, g_Vz );
		const GLuint lightPointLoc = glGetUniformLocation( m_OilPaintingShader, "lightPoint" );
		glUniform3f( lightPointLoc, g_Lx, g_Ly, g_Lz );
		const GLuint glossinessLoc = glGetUniformLocation( m_OilPaintingShader, "glossiness" );
		glUniform1f( glossinessLoc, g_Glossiness );
		const GLuint kdLoc = glGetUniformLocation( m_OilPaintingShader, "kd" );
		glUniform1f( kdLoc, g_kd );
		const GLuint ksLoc = glGetUniformLocation( m_OilPaintingShader, "ks" );
		glUniform1f( ksLoc, g_ks );
		const GLuint kaLoc = glGetUniformLocation( m_OilPaintingShader, "ka" );
		glUniform1f( kaLoc, g_ka );
		const GLuint lightIntensityLoc = glGetUniformLocation( m_OilPaintingShader, "lightIntensity" );
		glUniform1f( lightIntensityLoc, g_LightIntensity );
		const GLuint canvasScaleLoc = glGetUniformLocation( m_OilPaintingShader, "canvasScale" );
		glUniform1f( canvasScaleLoc, g_CanvasScale );
	}

	bool finalize()
	{
		glfwTerminate();
		return true;
	}
	
	template<typename T, int N>
	GLuint setupTexture( const Image<T, N>& in_image )
	{
		GLuint texture = 0; 
	  glGenTextures( 1, &texture );
	  glBindTexture( GL_TEXTURE_2D, texture );
  
	  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, in_image.getWidth(), in_image.getHeight(), 0, GL_RGBA, GL_FLOAT, in_image.getPtr() );
  
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );	
		
		m_Textures.insert( texture );
		
		return texture;
	}
	
	void useTexture( GLuint in_Texture )
	{
		m_StrokeTexture = in_Texture;	
	}
	
	void deleteTexture( GLuint in_Texture )
	{
		auto q = m_Textures.find( in_Texture );
		if( q != m_Textures.end() )
		{
			GLuint _tid = in_Texture;
			glDeleteTextures( 1, &_tid );
			m_Textures.erase( q );
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
	
	void clearRetinaAware()
	{
    glViewport( 0, 0, m_Width * g_FrameSize_WindowSize_Scale_x, m_Height * g_FrameSize_WindowSize_Scale_y );
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
  	
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * 2 );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteVertexArrays( 1, &vao );
	}
	
	void drawTextureFullFlip()
	{	
		float points[] = {
			-1.0f, -1.0f,
			-1.0f, 1.0f,
			1.0f,  -1.0f,
			-1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, -1.0f
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
	GLuint m_HeightShader;
	GLuint m_CombineHeightShader;
	GLuint m_ColorShader;
	GLuint m_OilPaintingShader;
	GLuint m_StrokeTexture;
	glm::mat4 m_MVP;
	
	Image<float, 4> m_Buffer;
	unsigned char* m_BufferBytes;
	
	std::set<GLuint> m_Textures;
};

Image<float, 4> g_Image;
StrokeData g_Strokes;
GLuint g_ColorTex = 0;
GLuint g_HeightTex = 0;
TexAuxData g_TexAux;
Renderer* g_Renderer;

void display_gui()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::IsWindowFocused();
	
	ImGui::Begin( "Control Panel" );
	
  if( ImGui::Button( "1. Load stroke data..." ) ) 
    ImGuiFileDialog::Instance()->OpenDialog( "CFx000", "Choose HDF5 File", ".h5", "." );
  if( ImGuiFileDialog::Instance()->Display( "CFx000" ) )
  {
    if( ImGuiFileDialog::Instance()->IsOk() )
    {
      std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
			int _width, _height;
			std::string _tex_fn;
			loadStrokes( filePathName, _width, _height, g_Strokes, _tex_fn, g_TexAux );
    }
    ImGuiFileDialog::Instance()->Close();
  }
	
  if( ImGui::Button( "2. Load color texture..." ) ) 
    ImGuiFileDialog::Instance()->OpenDialog( "CFx001", "Choose png File", ".png", "." );
  if( ImGuiFileDialog::Instance()->Display( "CFx001" ) )
  {
    if( ImGuiFileDialog::Instance()->IsOk() )
    {
      std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();			
			if( g_ColorTex != 0 ) g_Renderer->deleteTexture( g_ColorTex );
			Image<float, 4> color_tex_image;
			loadImagePng( filePathName, color_tex_image );
			g_ColorTex = g_Renderer->setupTexture( color_tex_image );
			std::cout << "color_tex: " << g_ColorTex << std::endl;
			
			if( g_TexAux.size(0) != color_tex_image.getWidth() || g_TexAux.size(1) != color_tex_image.getHeight() )
			{
				std::cout << "Mismatch found in color texture. Aux: " 
					<< g_TexAux.size(0) << ", " << g_TexAux.size(1) << "; tex size: " 
					<< color_tex_image.getWidth() << ", " << color_tex_image.getHeight() << "." << std::endl;
				std::cout << "Possible causes are 1) stroke data have not been loaded, or 2) a wrong texture has been specified." << std::endl;
			}
		}
    ImGuiFileDialog::Instance()->Close();
  }
	
  if( ImGui::Button( "3. Load height texture..." ) ) 
    ImGuiFileDialog::Instance()->OpenDialog( "CFx002", "Choose png File", ".png", "." );
  if( ImGuiFileDialog::Instance()->Display( "CFx002" ) )
  {
    if( ImGuiFileDialog::Instance()->IsOk() )
    {
      std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();			
			if( g_HeightTex != 0 ) g_Renderer->deleteTexture( g_HeightTex );
			Image<float, 4> height_tex_image;
			loadImagePng( filePathName, height_tex_image );
			g_HeightTex = g_Renderer->setupTexture( height_tex_image );
			std::cout << "color_tex: " << g_HeightTex << std::endl;
			
			if( g_TexAux.size(0) != height_tex_image.getWidth() || g_TexAux.size(1) != height_tex_image.getHeight() )
			{
				std::cout << "Mismatch found in color texture. Aux: " 
					<< g_TexAux.size(0) << ", " << g_TexAux.size(1) << "; tex size: " 
					<< height_tex_image.getWidth() << ", " << height_tex_image.getHeight() << "." << std::endl;
				std::cout << "Possible causes are 1) stroke data have not been loaded, or 2) a wrong texture has been specified." << std::endl;
			}	
		}
    ImGuiFileDialog::Instance()->Close();
  }
	
	
	float top_factor_exp = log2( g_CombineHeight_TopFactor );
	if( ImGui::SliderFloat( "Top factor exp", &top_factor_exp, -8, 0 ) )
	{
		g_CombineHeight_TopFactor = exp2( top_factor_exp );
		printParameters();
	}
	
	float additive_factor_exp = log2( g_CombineHeight_AdditiveFactor );
	if( ImGui::SliderFloat( "Additive factor exp", &additive_factor_exp, -8, 0 ) )
	{
		g_CombineHeight_AdditiveFactor = exp2( additive_factor_exp );
		printParameters();
	}
	
	float additive_log_factor_exp = log2( g_CombineHeight_AdditiveLogFactor );
	if( ImGui::SliderFloat( "Additive log factor exp", &additive_log_factor_exp, -8, 8 ) )
	{
		g_CombineHeight_AdditiveLogFactor = exp2( additive_log_factor_exp );
		printParameters();
	}
		
	float tex_stepx_exp = log2( g_TexStepX );
	if( ImGui::SliderFloat( "Tex step x exp", &tex_stepx_exp, -10, -3 ) )
	{
		g_TexStepX = exp2( tex_stepx_exp );
		printParameters();
	}
	
	float tex_stepy_exp = log2( g_TexStepY );
	if( ImGui::SliderFloat( "Tex step y exp", &tex_stepy_exp, -10, -3 ) )
	{
		g_TexStepY = exp2( tex_stepy_exp );
		printParameters();
	}
	
	float height_scale_exp = log2( g_HeightScale );
	if( ImGui::SliderFloat( "Height scale exp", &height_scale_exp, -12, 0 ) )
	{
		g_HeightScale = exp2( height_scale_exp );
		printParameters();
	}
	
	if( ImGui::SliderFloat( "Vz", &g_Vz, 0.01, 10.0 ) ) { printParameters(); }
	
	if( ImGui::SliderFloat( "Lx", &g_Lx, -10.0, 10.0 ) ) { printParameters(); }
	if( ImGui::SliderFloat( "Ly", &g_Ly, -10.0, 10.0 ) ) { printParameters(); }
	if( ImGui::SliderFloat( "Lz", &g_Lz, 0.01, 10.0 ) ) { printParameters(); }
	
	float glossiness_exp = log2( g_Glossiness );
	if( ImGui::SliderFloat( "Glossiness exp", &glossiness_exp, 0, 10 ) )
	{
		g_Glossiness = exp2( glossiness_exp );
		printParameters();
	}
	
	if( ImGui::SliderFloat( "kd", &g_kd, 0.0, 1.0 ) ) { printParameters(); }
	if( ImGui::SliderFloat( "ks", &g_ks, 0.0, 1.0 ) ) { printParameters(); }
	if( ImGui::SliderFloat( "ka", &g_ka, 0.0, 1.0 ) ) { printParameters(); }
	
	float intensity_exp = log2( g_LightIntensity );
	if( ImGui::SliderFloat( "Light intensity exp", &intensity_exp, 0, 10 ) )
	{
		g_LightIntensity = exp2( intensity_exp );
		printParameters();
	}
	
	if( ImGui::SliderFloat( "Canvas scale", &g_CanvasScale, 0.1, 2.0 ) ) { printParameters(); }
	
	if( ImGui::Button( "4. Save image..." ) )
		saveImagePngInvertY( "test_final.png", g_Image );	
	
	ImGui::End();
	
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

int main( int argc, char* argv[] )
{
	const int width = 1024;
	const int height = 1024;

	std::random_device rd;
	std::mt19937 mt(1);
	std::uniform_real_distribution<> dist(0.0, 1.0);
	
  if ( !glfwInit() )
    return -1;
	
	const char* glsl_version = "#version 150";
  glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
  glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 2 );
  glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
  glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
		
  GLFWwindow* window = glfwCreateWindow( width, height, "Renderer", NULL, NULL);
  if ( !window )
  {
    glfwTerminate();
    return -1;
  }
	
  glfwMakeContextCurrent( window );
	
  GLint dims[4] = { 0 };
  glGetIntegerv( GL_VIEWPORT, dims );
  g_FrameSize_WindowSize_Scale_x = double( dims[2] ) / double( width );
  g_FrameSize_WindowSize_Scale_y = double( dims[3] ) / double( height );

	glewExperimental = GL_TRUE;
  glewInit();
	
  const GLubyte* _renderer = glGetString( GL_RENDERER );
  const GLubyte* _version = glGetString( GL_VERSION );
  std::cout << "Renderer: " << _renderer << std::endl;
  std::cout << "OpenGL version supported: " << _version << std::endl;
	
	g_Renderer = new Renderer( width, height );
	if( !g_Renderer->initializeGL() )
		return -1;
		
	g_Image.init( width, height );
		
	FrameBufferObject* fbo_color = new FrameBufferObject( width, height );
	FrameBufferObject* fbo_height0 = new FrameBufferObject( width, height );
	FrameBufferObject* fbo_height = new FrameBufferObject( width, height );
	
	g_TexStepX = 1.0 / width;
	g_TexStepY = 1.0 / height;
	
	IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui_ImplGlfw_InitForOpenGL( window, true );
  ImGui_ImplOpenGL3_Init( glsl_version );
	
	const int n = 32;
	const float stroke_length = 0.2;
	const float stroke_width = 0.1;
	
	Eigen::Matrix4Xf colors; colors.resize( 4, n );
	for( int i=0; i<n; i++ )
	{
		colors.col(i) << float( dist(mt) ), float( dist(mt) ), float( dist(mt) ), 1.0f;
	}
	
	while ( !glfwWindowShouldClose( window ) )
	{
		glfwPollEvents();
		
		fbo_color->bind();
		g_Renderer->clear();
		g_Renderer->useTexture( g_ColorTex );
		g_Renderer->setTexAuxData( g_Renderer->colorShader(), g_TexAux );
		g_Renderer->drawColor( g_Strokes );	

		fbo_color->unbind();
	
		fbo_height0->bind();
		g_Renderer->clear();
		g_Renderer->useTexture( g_HeightTex );
		g_Renderer->setTexAuxData( g_Renderer->heightShader(), g_TexAux );
		g_Renderer->drawHeight( g_Strokes );
		
		fbo_height0->unbind();

		fbo_height->bind();
		g_Renderer->clear();
		glUseProgram( g_Renderer->combineHeightShader() );
		const GLuint topDivisorLoc = glGetUniformLocation( g_Renderer->combineHeightShader(), "top_factor" );
		glUniform1f( topDivisorLoc, g_CombineHeight_TopFactor );
		const GLuint additiveDivisorLoc = glGetUniformLocation( g_Renderer->combineHeightShader(), "additive_factor" );
		glUniform1f( additiveDivisorLoc, g_CombineHeight_AdditiveFactor );
		const GLuint additiveLogFactorLoc = glGetUniformLocation( g_Renderer->combineHeightShader(), "additive_log_factor" );
		glUniform1f( additiveLogFactorLoc, g_CombineHeight_AdditiveLogFactor );
		
		glDisable( GL_BLEND );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, fbo_height0->colorBuffer() );
		g_Renderer->drawTextureFull();
	
		fbo_height->unbind();
	
		g_Renderer->clearRetinaAware();
		glUseProgram( g_Renderer->oilPaintingShader() );
		const GLuint texStepLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "texStep" );
		glUniform2f( texStepLoc, g_TexStepX, g_TexStepY );
		const GLuint heightScaleLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "heightScale" );
		glUniform1f( heightScaleLoc, g_HeightScale );
		const GLuint viewPointLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "viewPoint" );
		glUniform3f( viewPointLoc, 0.0, 0.0, g_Vz );
		const GLuint lightPointLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "lightPoint" );
		glUniform3f( lightPointLoc, g_Lx, g_Ly, g_Lz );
		const GLuint glossinessLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "glossiness" );
		glUniform1f( glossinessLoc, g_Glossiness );
		const GLuint kdLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "kd" );
		glUniform1f( kdLoc, g_kd );
		const GLuint ksLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "ks" );
		glUniform1f( ksLoc, g_ks );
		const GLuint kaLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "ka" );
		glUniform1f( kaLoc, g_ka );
		const GLuint lightIntensityLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "lightIntensity" );
		glUniform1f( lightIntensityLoc, g_LightIntensity );
		const GLuint canvasScaleLoc = glGetUniformLocation( g_Renderer->oilPaintingShader(), "canvasScale" );
		glUniform1f( canvasScaleLoc, g_CanvasScale );
		glDisable( GL_BLEND );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, fbo_color->colorBuffer() );
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture( GL_TEXTURE_2D, fbo_height->colorBuffer() );
	
		g_Renderer->drawTextureFullFlip();
		
		g_Renderer->readBuffer( g_Image );
		
		display_gui();
		
		glfwSwapBuffers( window );
	}
	
	ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
	
	delete fbo_color;
	delete fbo_height0;
	delete fbo_height;
	g_Renderer->finalize();
	delete g_Renderer;
	
	return 0;
}
