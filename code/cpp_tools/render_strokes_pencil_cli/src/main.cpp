// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: render_strokes_pencil_cli/src/main.cpp
// Maintainer: Yonghao Yue
//
// Description:
// This file implements the final rendering step for strokes in a pencil 
// style (to account for lighting, etc).
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

struct TexAuxData
{
  Eigen::Vector2i size;
  int single_w;
  int single_h;
	float min_tex_ratio;
  int num_textures;
  int length_mipmap_level;
};

void showTexAuxData( const TexAuxData& in_data )
{
	std::cout << "texauxdata: " << std::endl;
	std::cout << in_data.size.transpose() << std::endl;
	std::cout << in_data.single_w << ", " << in_data.single_h << ", " << in_data.min_tex_ratio << std::endl;
	std::cout << in_data.num_textures << ", " << in_data.length_mipmap_level << std::endl;
}

class Stroke
{
	Stroke();
public:
	Stroke( const std::vector<Eigen::Vector2f>& in_center_line, const float in_width, const float* in_color4 )
		: m_Width( in_width ), m_num_tris( 0 )
	{
		for( int i=0; i<4; i++ )
			m_Color(i) = in_color4[i];
		
		m_CenterLine.resize( 2, in_center_line.size() );
		m_T.resize( 2, in_center_line.size() );
		m_B.resize( 2, in_center_line.size() );
		m_ArcParameter.resize( in_center_line.size() );
		
		for( int i=0; i<in_center_line.size(); i++ )
		{
			m_CenterLine.col(i) = in_center_line[i];
		}
		
		m_T.col(0) = ( in_center_line[1] - in_center_line[0] ).normalized();
		m_ArcParameter(0) = 0.0;
		for( int i=1; i<in_center_line.size()-1; i++ )
		{
			const Eigen::Vector2f E0 = ( in_center_line[i+1] - in_center_line[i] ).normalized();
			const Eigen::Vector2f _E1 = in_center_line[i] - in_center_line[i-1];
			const float E1_norm = _E1.norm();
			m_ArcParameter(i) = m_ArcParameter(i-1) + E1_norm;
			const Eigen::Vector2f E1 = _E1 / E1_norm;
			m_T.col(i) = ( 0.5 * ( E0 + E1 ) ).normalized();
		}
		
		const Eigen::Vector2f _E1 = in_center_line[ in_center_line.size()-1 ]  - in_center_line[ in_center_line.size()-2 ];
		const float E1_norm = _E1.norm();
		m_T.col(in_center_line.size()-1) = _E1 / E1_norm;
		m_ArcParameter(in_center_line.size()-1) = m_ArcParameter(in_center_line.size()-2) + E1_norm;
		
		for( int i=0; i<in_center_line.size(); i++ )
		{
			m_B.col(i).x() = m_T.col(i).y();
			m_B.col(i).y() = -m_T.col(i).x();
			m_ArcParameter(i) /= m_ArcParameter(in_center_line.size()-1);
		}
		
		m_num_tris = ( in_center_line.size() - 1 ) * 2;
		m_vertices.resize( m_num_tris * 3 * 2 );
		m_uvs.resize( m_num_tris * 3 * 2 );
		m_ts.resize( m_num_tris * 3 * 2 );
		
		// center line pos coord: [0, 1]^2
		// vertices coord: [-1, 1]^2
		
		Eigen::Vector2f P0 = m_CenterLine.col(0) - 0.5 * m_Width * m_B.col(0);
		Eigen::Vector2f P1 = m_CenterLine.col(0) + 0.5 * m_Width * m_B.col(0);
		
		for( int i=0; i<in_center_line.size()-1; i++ )
		{
			const Eigen::Vector2f P2 = m_CenterLine.col(i+1) - 0.5 * m_Width * m_B.col(i+1);
			const Eigen::Vector2f P3 = m_CenterLine.col(i+1) + 0.5 * m_Width * m_B.col(i+1);
			
			m_vertices( 2*i*3*2      ) = P0.x(); m_vertices( 2*i*3*2 + 1  ) = P0.y();
			m_vertices( 2*i*3*2 + 2  ) = P1.x(); m_vertices( 2*i*3*2 + 3  ) = P1.y();
			m_vertices( 2*i*3*2 + 4  ) = P3.x(); m_vertices( 2*i*3*2 + 5  ) = P3.y();
			m_vertices( 2*i*3*2 + 6  ) = P0.x(); m_vertices( 2*i*3*2 + 7  ) = P0.y();
			m_vertices( 2*i*3*2 + 8  ) = P3.x(); m_vertices( 2*i*3*2 + 9  ) = P3.y();
			m_vertices( 2*i*3*2 + 10 ) = P2.x(); m_vertices( 2*i*3*2 + 11 ) = P2.y();
			
			m_uvs( 2*i*3*2      ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 1  ) = 0.0;
			m_uvs( 2*i*3*2 + 2  ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 3  ) = 1.0;
			m_uvs( 2*i*3*2 + 4  ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 5  ) = 1.0;
			m_uvs( 2*i*3*2 + 6  ) = m_ArcParameter(i);   m_uvs( 2*i*3*2 + 7  ) = 0.0;
			m_uvs( 2*i*3*2 + 8  ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 9  ) = 1.0;
			m_uvs( 2*i*3*2 + 10 ) = m_ArcParameter(i+1); m_uvs( 2*i*3*2 + 11 ) = 0.0;
			
			m_ts( 2*i*3*2      ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 1  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 2  ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 3  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 4  ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 5  ) = m_T.col(i+1).y();
			m_ts( 2*i*3*2 + 6  ) = m_T.col(i).x();   m_ts( 2*i*3*2 + 7  ) = m_T.col(i).y();
			m_ts( 2*i*3*2 + 8  ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 9  ) = m_T.col(i+1).y();
			m_ts( 2*i*3*2 + 10 ) = m_T.col(i+1).x(); m_ts( 2*i*3*2 + 11 ) = m_T.col(i+1).y();
			
			P0 = P2;
			P1 = P3;
		}
		
		// BB coord: [0, 1]^2
		
		m_BBMin(0) = 1.0e33; m_BBMin(1) = 1.0e33;
		m_BBMax(0) = -1.0e33; m_BBMax(1) = -1.0e33;
		
		for( int i=0; i<m_num_tris*3; i++ )
		{
			m_BBMin(0) = std::min<float>( m_BBMin(0), m_vertices[ 2*i ] );
			m_BBMin(1) = std::min<float>( m_BBMin(1), m_vertices[ 2*i+1 ] );
			m_BBMax(0) = std::max<float>( m_BBMax(0), m_vertices[ 2*i ] );
			m_BBMax(1) = std::max<float>( m_BBMax(1), m_vertices[ 2*i+1 ] );
		}
		
		// [0, 1]^2 -> [-1, 1]2^
		for( int i=0; i<m_num_tris * 3 * 2; i++ )
			m_vertices(i) = 2.0 * m_vertices(i) - 1.0;
	}
	
	~Stroke()
	{}
	
	int numTris() const
	{
		return m_num_tris;
	}
	
	const float* vertices() const
	{
		return m_vertices.data();
	}
	
	const float* uvs() const
	{
		return m_uvs.data();
	}
	
	const float* ts() const
	{
		return m_ts.data();
	}
	
	void color( float out_color[4] ) const
	{
		for( int i=0; i<4; i++ )
			out_color[i] = m_Color(i);
	}
	
	const Eigen::Vector2f& BBMin() const
	{
		return m_BBMin;
	}
	
	const Eigen::Vector2f& BBMax() const
	{
		return m_BBMax;
	}
	
	const Eigen::Matrix2Xf& centerLine() const
	{
		return m_CenterLine;
	}
	
	float width() const
	{
		return m_Width;
	}
	
	const Eigen::Vector4f& color() const
	{
		return m_Color;
	}
		
private:
	Eigen::Matrix2Xf m_CenterLine;
	float m_Width;
	Eigen::Vector4f m_Color;
	
	Eigen::VectorXf m_vertices;
	Eigen::VectorXf m_uvs;
	Eigen::VectorXf m_ts;
	int m_num_tris;
	Eigen::Vector2f m_BBMin;
	Eigen::Vector2f m_BBMax;
	Eigen::Matrix2Xf m_T;
	Eigen::Matrix2Xf m_B;
	Eigen::VectorXf m_ArcParameter;
};

struct StrokeData
{
	std::vector<Stroke> strokes;
	std::vector<int> indices;
	std::vector<Eigen::Vector2i> tex_info_int;
	std::vector<Eigen::Vector2f> tex_info_float;
	
	void clear()
	{
		strokes.clear();
		indices.clear();
		tex_info_int.clear();
		tex_info_float.clear();
	}
};

void loadStrokes( const std::string& in_h5_fn, int& out_width, int& out_height, StrokeData& out_strokes, 
	std::string& out_texture_filename, TexAuxData& out_aux )
{
	HDF5File h5( in_h5_fn, HDF5AccessType::READ_ONLY );
	Eigen::Vector2i resolution;
	h5.readVector( "strokes", "resolution", resolution );
	out_width = resolution(0);
	out_height = resolution(1);
	h5.readString( "strokes", "texture_filename", out_texture_filename );
	
	Eigen::Vector3i tex_aux_shape;
	h5.readVector( "strokes/tex_aux", "shape", tex_aux_shape );
	out_aux.size(0) = tex_aux_shape(1); out_aux.size(1) = tex_aux_shape(0);
	std::cout << "out_aux.shape: " << out_aux.size.transpose() << std::endl;
	
	Eigen::Vector3i tex_aux_single_size;
	h5.readVector( "strokes/tex_aux", "single_size", tex_aux_single_size );
	out_aux.single_w = tex_aux_single_size(0); out_aux.single_h = tex_aux_single_size(1); 
	std::cout << "out_aux.single: " << out_aux.single_w << ", " << out_aux.single_h << std::endl;
	
	Eigen::Matrix<int,1,1> tex_aux_num_textures;
	h5.readVector( "strokes/tex_aux", "num_textures", tex_aux_num_textures );
	out_aux.num_textures = tex_aux_num_textures(0);
	std::cout << "out_aux.num_textures: " << out_aux.num_textures << std::endl;
	
	Eigen::Matrix<int,1,1> tex_aux_length_mipmap_level;
	h5.readVector( "strokes/tex_aux", "length_mipmap_level", tex_aux_length_mipmap_level );
	out_aux.length_mipmap_level = tex_aux_length_mipmap_level(0);
	std::cout << "out_aux.length_mipmap_level: " << out_aux.length_mipmap_level << std::endl;
	
	Eigen::Matrix<float,1,1> tex_aux_min_tex_ratio;
	h5.readVector( "strokes/tex_aux", "min_tex_ratio", tex_aux_min_tex_ratio );
	out_aux.min_tex_ratio = tex_aux_min_tex_ratio(0);
	std::cout << "out_aux.min_tex_ratio: " << out_aux.min_tex_ratio << std::endl;
	
	Eigen::MatrixXf stroke_tex_info;
	h5.readMatrix( "strokes", "stroke_tex_info", stroke_tex_info );
	
	out_strokes.tex_info_int.resize( stroke_tex_info.rows() );
	out_strokes.tex_info_float.resize( stroke_tex_info.rows() );
	
	for( int i=0; i<stroke_tex_info.rows(); i++ )
	{
		out_strokes.tex_info_int[i][0] = int( floor( 0.5 + stroke_tex_info( i, 0 ) ) );
		out_strokes.tex_info_int[i][1] = int( floor( 0.5 + stroke_tex_info( i, 1 ) ) );
		out_strokes.tex_info_float[i][0] = stroke_tex_info( i, 2 );
		out_strokes.tex_info_float[i][1] = stroke_tex_info( i, 3 );
	}
	
	Eigen::VectorXi stroke_indices;
	h5.readVector( "strokes", "stroke_indices", stroke_indices );
	out_strokes.indices.resize( stroke_indices.size() );
	for( int i=0; i<stroke_indices.size(); i++ )
		out_strokes.indices[i] = stroke_indices(i);
	
	Eigen::Matrix<int,1,1> num_strokes;
	h5.readVector( "strokes", "num_strokes", num_strokes );
	
	out_strokes.strokes.clear();
	for( int i=0; i<num_strokes(0); i++ )
	{
		std::string group_name = std::string( "strokes/stroke" ) + std::to_string(i);
		Eigen::Matrix2Xf _centerLine;
		h5.readMatrix( group_name.c_str(), "center_line", _centerLine );
		Eigen::Matrix<float,1,1> _width;
		h5.readVector( group_name.c_str(), "width", _width );
		Eigen::Vector4f _color;
		h5.readVector( group_name.c_str(), "color", _color );
		
		std::vector<Eigen::Vector2f> center_line_vect; center_line_vect.resize( _centerLine.cols() );
		for( int j=0; j<_centerLine.cols(); j++ )
			center_line_vect[j] = _centerLine.col(j);
		
		out_strokes.strokes.emplace_back( center_line_vect, _width(0), _color.data() );
	}
}

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

struct LightingParams
{
	float tex_step_x;
	float tex_step_y;
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
	float pencil_factor;
	float canvas_scale;
};

void GetShaderInfoLog( GLuint in_shader )
{
	GLsizei bufSize;
	glGetShaderiv( in_shader, GL_INFO_LOG_LENGTH , &bufSize) ;

	if( bufSize > 1 ) 
	{
		GLchar *infoLog = ( GLchar * )malloc( bufSize );

		if( infoLog != NULL ) 
		{
			GLsizei length;

			glGetShaderInfoLog( in_shader, bufSize, &length, infoLog );
			std::cout << "InfoLog: " << infoLog << std::endl;
			free( infoLog );
		}
		else
			std::cout << "Could not allocate InfoLog buffer" << std::endl;
	}
}

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
		
		compileLightingShader();
		compilePencilShader();
			
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
		return true;
	}
	
	void compileLightingShader()
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
		"  vec4 col = vec4( 1.0, 1.0, 1.0, 1.0 );\n"
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
  
	  m_LightingShader = glCreateProgram();
	  glAttachShader( m_LightingShader, fs );
	  glAttachShader( m_LightingShader, vs );
	  glLinkProgram( m_LightingShader );
		
		std::cout << "m_LightingShader" << std::endl;
		GetShaderInfoLog( m_LightingShader );
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0 + 2.0 * double( m_Height ) / max_res, -1.0, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
				
		glUseProgram( m_LightingShader );
    const GLuint MatrixID = glGetUniformLocation( m_LightingShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint heightTexLoc = glGetUniformLocation( m_LightingShader, "heightTex" );
		glUniform1i( heightTexLoc, 1 );
		const GLuint texStepLoc = glGetUniformLocation( m_LightingShader, "texStep" );
		glUniform2f( texStepLoc, m_LightingParams.tex_step_x, m_LightingParams.tex_step_y );
		const GLuint heightScaleLoc = glGetUniformLocation( m_LightingShader, "heightScale" );
		glUniform1f( heightScaleLoc, m_LightingParams.height_scale );
		const GLuint viewPointLoc = glGetUniformLocation( m_LightingShader, "viewPoint" );
		glUniform3f( viewPointLoc, 0.0, 0.0, m_LightingParams.vz );
		const GLuint lightPointLoc = glGetUniformLocation( m_LightingShader, "lightPoint" );
		glUniform3f( lightPointLoc, m_LightingParams.lx, m_LightingParams.ly, m_LightingParams.lz );
		const GLuint glossinessLoc = glGetUniformLocation( m_LightingShader, "glossiness" );
		glUniform1f( glossinessLoc, m_LightingParams.glossiness );
		const GLuint kdLoc = glGetUniformLocation( m_LightingShader, "kd" );
		glUniform1f( kdLoc, m_LightingParams.kd );
		const GLuint ksLoc = glGetUniformLocation( m_LightingShader, "ks" );
		glUniform1f( ksLoc, m_LightingParams.ks );
		const GLuint kaLoc = glGetUniformLocation( m_LightingShader, "ka" );
		glUniform1f( kaLoc, m_LightingParams.ka );
		const GLuint lightIntensityLoc = glGetUniformLocation( m_LightingShader, "lightIntensity" );
		glUniform1f( lightIntensityLoc, m_LightingParams.light_intensity );
		const GLuint canvasScaleLoc = glGetUniformLocation( m_LightingShader, "canvasScale" );
		glUniform1f( canvasScaleLoc, m_LightingParams.canvas_scale );
	}
	
	void compilePencilShader()
	{
	  const char* vertex_shader =
	  "#version 400\n"
	  "in vec2 vp;\n"
		"layout(location = 1) in vec2 uv;\n"
		"layout(location = 2) in vec2 tangent;\n"
	  "uniform mat4 MVP;\n"
		"out vec2 fragmentUV;\n"
		"out vec2 T;\n"
		"out vec2 worldUV;\n"
	  "void main() {\n"
	  "  gl_Position = MVP * vec4(vp, 0.0, 1.0);\n"
		"  fragmentUV = uv;\n"
		"  T = tangent;\n"
		"  worldUV = vec2( ( 1.0 + gl_Position.x ) * 0.5, ( 1.0 - gl_Position.y ) * 0.5 );\n"
	  "}";
		
	  const char* fragment_shader =
	  "#version 400\n"
		"in vec2 fragmentUV;\n"
		"in vec2 T;\n"
		"in vec2 worldUV;\n"
		"uniform vec4 strokeColor;\n"
		"uniform vec2 singleTexSize;\n"
		"uniform vec2 texSize;\n"
		"uniform int tex_id;\n"
		"uniform int mipmap_level;\n"
		"uniform float o_s;\n"
		"uniform float o_e;\n"
		"uniform sampler2D colorTex;\n"
		"uniform sampler2D heightTex;\n"
		"uniform vec2 texStep;\n"
		"uniform float heightScale;\n"
		"uniform float pencilFactor;\n"
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
		"  float h_c = heightScale * texture(heightTex, worldUV).x;\n"
		"  float h_x = heightScale * texture(heightTex, vec2(worldUV.x+texStep.x, worldUV.y) ).x;\n"
		"  float h_y = heightScale * texture(heightTex, vec2(worldUV.x, worldUV.y-texStep.y) ).x;\n"
		"  vec3 _n = vec3( -( h_x - h_c ) / texStep.x, -( h_y - h_c ) / texStep.y, 1.0 );\n"
		"  vec3 n = normalize( _n );\n"
		"  vec3 _t = normalize( vec3( T.x, T.y, 0.0 ) );\n"
		"  float _dot = dot( _t, n );\n"
		"  float _theta = asin( -_dot );\n"
		"  float c = ( pencilFactor + tan( _theta ) ) / pencilFactor;\n"
		"  vec4 tex_color = texture(colorTex, warped_uv);\n"
		"  finalColor = strokeColor * vec4( 1.0, 1.0, 1.0, c * tex_color.w );\n"
		"}";
				
	  const GLuint vs = glCreateShader( GL_VERTEX_SHADER );
	  glShaderSource( vs, 1, &vertex_shader, NULL );
	  glCompileShader( vs );
	  const GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
	  glShaderSource( fs, 1, &fragment_shader, NULL );
	  glCompileShader( fs );
  
	  m_PencilShader = glCreateProgram();
	  glAttachShader( m_PencilShader, fs );
	  glAttachShader( m_PencilShader, vs );
	  glLinkProgram( m_PencilShader );
		
		std::cout << "m_PencilShader" << std::endl;
		GetShaderInfoLog( m_PencilShader );
		
		int success = 0;
		glGetProgramiv( m_PencilShader, GL_LINK_STATUS, &success );
		if (!success) 
		{
			char infoLog[512];
			glGetProgramInfoLog( m_PencilShader, 512, NULL, infoLog );
			std::cout << "ERROR: Shader linking failed" << std::endl;
			std::cout << "\t" << infoLog << std::endl;
		}
		
		const int max_res = std::max<int>( m_Width, m_Height );
    glm::mat4 Projection = glm::ortho( -1.0, -1.0 + 2.0 * double( m_Width ) / max_res, -1.0, -1.0 + 2.0 * double( m_Height ) / max_res, 0.1, 100.0 );
    glm::mat4 View = glm::lookAt( glm::vec3( 0.0, 0.0, 2.0 ), glm::vec3( 0.0, 0.0, 0.0 ), glm::vec3( 0.0, 1.0, 0.0 ) );
    glm::mat4 Model = glm::mat4(1.0f);
    m_MVP = Projection * View * Model;
		
		glUseProgram( m_PencilShader );
    const GLuint MatrixID = glGetUniformLocation( m_PencilShader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &m_MVP[0][0] );
		const GLuint colorTexLoc = glGetUniformLocation( m_PencilShader, "colorTex" );
		glUniform1i( colorTexLoc, 0 );
		const GLuint heightTexLoc = glGetUniformLocation( m_PencilShader, "heightTex" );
		glUniform1i( heightTexLoc, 1 );
		const GLuint texStepLoc = glGetUniformLocation( m_PencilShader, "texStep" );
		glUniform2f( texStepLoc, m_LightingParams.tex_step_x, m_LightingParams.tex_step_y );
		const GLuint heightScaleLoc = glGetUniformLocation( m_PencilShader, "heightScale" );
		glUniform1f( heightScaleLoc, m_LightingParams.height_scale );
		const GLuint pencilFactorLoc = glGetUniformLocation( m_PencilShader, "pencilFactor" );
		glUniform1f( pencilFactorLoc, m_LightingParams.pencil_factor );
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
  
	  GLuint tbuffer = 0;
	  glGenBuffers(1, &tbuffer);
	  glBindBuffer(GL_ARRAY_BUFFER, tbuffer);
	  glBufferData(GL_ARRAY_BUFFER, 3 * 2 * in_stroke.numTris() * sizeof(float), in_stroke.ts(), GL_STATIC_DRAW);
  
	  GLuint vao = 0;
	  glGenVertexArrays( 1, &vao );
	  glBindVertexArray( vao );
	  glEnableVertexAttribArray(0);
	  glBindBuffer( GL_ARRAY_BUFFER, vbo );
	  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, NULL );
		
	  glEnableVertexAttribArray(1);
	  glBindBuffer( GL_ARRAY_BUFFER, uvbuffer );
	  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, NULL );
  	
	  glEnableVertexAttribArray(2);
	  glBindBuffer( GL_ARRAY_BUFFER, tbuffer );
	  glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 0, NULL );
  	
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, m_StrokeTexture );
		
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 * in_stroke.numTris() );
		
		glDeleteBuffers( 1, &vbo );
		glDeleteBuffers( 1, &uvbuffer );
		glDeleteBuffers( 1, &tbuffer );
		glDeleteVertexArrays( 1, &vao );
	}
	
	void _draw( GLuint in_Shader, const StrokeData& in_strokes )
	{
		for( int i=0; i<in_strokes.strokes.size(); i++ )
		{
			_draw( in_Shader, in_strokes.strokes[i], in_strokes.tex_info_int[i], in_strokes.tex_info_float[i] );
		}
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
	
	GLuint lightingShader()
	{
		return m_LightingShader;
	}
	
	GLuint pencilShader()
	{
		return m_PencilShader;
	}
		
private:
	int m_Width;
	int m_Height;
	GLFWwindow* m_Window;
	GLuint m_LightingShader;
	GLuint m_PencilShader;
	GLuint m_StrokeTexture;
	glm::mat4 m_MVP;
	
	Image<float, 4> m_Buffer;
	unsigned char* m_BufferBytes;
	
	std::set<GLuint> m_GeneratedTextures;
	
	LightingParams m_LightingParams;
};

void compute_tex_settings( const Eigen::Vector2i& in_texture_size, const int in_num_textures, const int in_texture_length_mipmap_level,
	TexAuxData& out_tex_aux_data )
{
  int num_tex_units = 0;
  int unit_size = 1;
  for( int i=0; i<in_texture_length_mipmap_level; i++ )
	{
    num_tex_units += unit_size;
    unit_size *= 2;
	}
	
	out_tex_aux_data.size = in_texture_size;
	out_tex_aux_data.single_w = int( in_texture_size(0) / num_tex_units );
  out_tex_aux_data.single_h = int( in_texture_size(1) / in_num_textures );
	out_tex_aux_data.min_tex_ratio = out_tex_aux_data.single_w / out_tex_aux_data.single_h;
  out_tex_aux_data.num_textures = in_num_textures;
  out_tex_aux_data.length_mipmap_level = in_texture_length_mipmap_level;
  
  std::cout << "tex_single_size: " << out_tex_aux_data.single_w << "(w), " << out_tex_aux_data.single_h << "(h)" << std::endl;
  
	if( out_tex_aux_data.single_w * num_tex_units != in_texture_size(0) || out_tex_aux_data.single_h * in_num_textures != in_texture_size(1) )
    std::cout << "Illegal texture size. Please double check the texture content, as well as the settings for num_textures and texture_length_mipmap_level." << std::endl;      
}

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
		("paper_texture_filename", boost::program_options::value<std::string>()->default_value("../../textures/paper.png"), "paper_texture_filename" )
		("num_textures", boost::program_options::value<int>()->default_value(1), "num_textures" )
		("texture_length_mipmap_level", boost::program_options::value<int>()->default_value(1), "texture_length_mipmap_level" )
		("stroke_data_filename_template", boost::program_options::value<std::string>()->default_value("stroke_data/stroke_%03d.h5"), "stroke_data_filename_template" )
		("out_final_filename_template", boost::program_options::value<std::string>()->default_value("final/final_%03d.png"), "out_final_filename_template" )
		("undercoat_filename_template", boost::program_options::value<std::string>()->default_value("color/color_%03d.png"), "undercoat_filename_template" )
		("mask_file_template", boost::program_options::value<std::string>()->default_value(""), "mask_file_template" )
	  ("tex_step_x", boost::program_options::value<float>()->default_value( exp2( -7.953 ) ), "tex_step_x")
		("tex_step_y", boost::program_options::value<float>()->default_value( exp2( -7.953 ) ), "tex_step_y")
		("height_scale", boost::program_options::value<float>()->default_value( 0.00075 ), "height_scale")
		("vz", boost::program_options::value<float>()->default_value( 3.2 ), "vz")
		("lx", boost::program_options::value<float>()->default_value( 0.0 ), "lx")
		("ly", boost::program_options::value<float>()->default_value( 1.2 ), "ly")
		("lz", boost::program_options::value<float>()->default_value( 3.2 ), "lz")
		("glossiness", boost::program_options::value<float>()->default_value( exp2( 8.130 ) ), "glossiness")
		("kd", boost::program_options::value<float>()->default_value( 0.24 ), "kd")
		("ks", boost::program_options::value<float>()->default_value( 0.01 ), "ks")
		("ka", boost::program_options::value<float>()->default_value( 0.29 ), "ka")
		("light_intensity", boost::program_options::value<float>()->default_value( exp2( 1.53 ) ), "light_intensity")
		("pencil_factor", boost::program_options::value<float>()->default_value( 0.5 ), "pencil_factor")
		("canvas_scale", boost::program_options::value<float>()->default_value( 0.4 ), "canvas_scale")
		("log_file_name", boost::program_options::value<std::string>()->default_value("stroke/log.txt"), "log_file_name" );
	
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
	const std::string paper_texture_filename = vm["paper_texture_filename"].as<std::string>();
	std::cout << "paper_texture_filename: " << paper_texture_filename << std::endl;
	
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
	lighting_params.tex_step_x = vm["tex_step_x"].as<float>();
	lighting_params.tex_step_y = vm["tex_step_y"].as<float>();
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
	lighting_params.pencil_factor = vm["pencil_factor"].as<float>();
	lighting_params.canvas_scale = vm["canvas_scale"].as<float>();	
	
	std::chrono::steady_clock::time_point _begin, _end;
	
	StrokeRenderer* renderer = new StrokeRenderer( width, height, lighting_params );
	if( !renderer->initializeGL() )
		return -1;
		
	Image<float, 4> image;
	image.init( width, height );
	
	Image<float, 4> color_texture;
	loadImagePng( color_texture_filename, color_texture );
	Image<float, 4> paper_texture;
	loadImagePng( paper_texture_filename, paper_texture );
	
	int tex_single_w, tex_single_h;
	float min_tex_ratio;
	
	TexAuxData tex_aux_data;
	compute_tex_settings( Eigen::Vector2i{ color_texture.getWidth(), color_texture.getHeight() }, num_textures, texture_length_mipmap_level, tex_aux_data );

  std::cout << "tex_single_w: " << tex_aux_data.single_w << std::endl;
  std::cout << "tex_single_h: " << tex_aux_data.single_h << std::endl;
	
	GLuint color_texture_index = renderer->setupStrokeTexture( color_texture );
	GLuint paper_texture_index = renderer->setupStrokeTexture( paper_texture );
	
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
		std::string undercoat_filename = undercoat_path;
		
		std::string mask_filename = "";
		if( mask_file_template != "" )
		{
			snprintf( buf, buf_size, mask_file_template.c_str(), frame_idx ); std::filesystem::path mask_path = stroke_dir; mask_path /= buf;
			mask_filename = mask_path;
		}
		
		snprintf( buf, buf_size, stroke_data_filename_template.c_str(), frame_idx ); std::filesystem::path stroke_data_path = stroke_dir; stroke_data_path /= buf;
		std::string stroke_data_filename = stroke_data_path;
		
		snprintf( buf, buf_size, out_final_filename_template.c_str(), frame_idx ); std::filesystem::path out_final_path = stroke_dir; out_final_path /= buf;
		std::string out_final_filename = out_final_path; 
		
		int _width, _height;
		std::string _texture_filename;
		TexAuxData _aux;
		std::cout << "stroke_data_filename: " << stroke_data_filename << std::endl;
		loadStrokes( stroke_data_filename, _width, _height, strokes, _texture_filename, _aux ); 
		if( color_texture.getWidth() != _aux.size(0) || color_texture.getHeight() != _aux.size(1) )	
			std::cout << "WARNING!!!: mismatch in image size between color_texture and texture associated with stroke data..." << std::endl;
		
		renderer->clear();
		glUseProgram( renderer->lightingShader() );
		glDisable( GL_BLEND );
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture( GL_TEXTURE_2D, paper_texture_index );
		renderer->drawTextureFullFlip();
		
		glUseProgram( renderer->pencilShader() );
		renderer->setTexAuxData( renderer->pencilShader(), tex_aux_data );
		
    glEnable( GL_BLEND );
    glBlendEquationSeparate( GL_FUNC_ADD, GL_FUNC_ADD );
    glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
		
		renderer->useTexture( color_texture_index );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, color_texture_index );
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture( GL_TEXTURE_2D, paper_texture_index );
		
		renderer->_draw( renderer->pencilShader(), strokes );
		
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
