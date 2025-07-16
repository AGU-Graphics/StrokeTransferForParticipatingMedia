// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: render_surface_features_cli/src/main.cpp
// Maintainer: Hideki Todo and Yonghao Yue
//
// Description:
// Given a surface object (via obj file) and camera information (via json file),
// this code computes the normal and curvature features using libigl, as well as
// the relative velocity feature(s).
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
#include <Eigen/LU>

#include <iostream>
#include <filesystem>
#include <set>
#include <cmath>

#include <lib/image/image.h>
#include <lib/image/imageioutil.h>
#include <lib/image/imageiopng.h>
#include <HDF5File.h>

#include <random>
#include <chrono>
#include <cmath>

#include <boost/program_options.hpp>

#include <lib/strokes/framebufferobject.h>

#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <boost/json.hpp>

#include <iostream>
#include <typeinfo>

class SurfaceRenderer
{
	SurfaceRenderer();
public:
	SurfaceRenderer( const int in_width, const int in_height )
		: m_Width( in_width ), m_Height( in_height ), m_Window( nullptr ), m_BufferBytes( nullptr )
	{
		m_Buffer.init( in_width, in_height );
		m_BufferBytes = (unsigned char*)malloc( sizeof(unsigned char) * in_width * in_height * 4 );
	}
		
	~SurfaceRenderer()
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
		
	  m_Window = glfwCreateWindow( m_Width, m_Height, "Surface Renderer", NULL, NULL);
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
	  "in vec4 vp;\n"
		"layout(location = 1) in vec4 color;\n"
	  "uniform mat4 MVP;\n"
			"out vec4 c;\n"
	  "void main() {\n"
	  "  gl_Position = MVP * vp;\n"
		" c = color;\n"
	  "}";
  
	  const char* fragment_shader =
	  "#version 400\n"
		"in vec4 c;\n"
		"out vec4 finalColor;\n"
	  "void main() {\n"
		"  finalColor = c;\n"
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
	}
	
	void setMatrix( GLuint in_Shader, glm::mat4& in_mat )
	{
		glUseProgram( in_Shader );
    const GLuint MatrixID = glGetUniformLocation( in_Shader, "MVP" );
    glUniformMatrix4fv( MatrixID, 1, GL_FALSE, &in_mat[0][0] );
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
		
		m_GeneratedTextures.insert( texture );
		
		return texture;
	}
	
	void useTexture( GLuint in_Texture )
	{
		m_Texture = in_Texture;	
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
	
	GLuint computeVBO( const Eigen::MatrixXf& in_Data, const Eigen::MatrixXi& in_F )
	{
		float* data = (float*)malloc( sizeof( float ) * 4 * 3 * in_F.rows() );
		for( int i=0; i<in_F.rows(); i++ )
		{
			for( int k=0; k<3; k++ )
			{
				for( int s=0; s<in_Data.cols(); s++ )
					data[ (i*3+k)*4+s ] = in_Data.row( in_F.row(i)(k) )(s);
				for( int s=in_Data.cols(); s<4; s++ )
					data[ (i*3+k)*4+s ] = 1.0;
			}
		}
		
	  GLuint out_vbo = 0;
	  glGenBuffers( 1, &out_vbo );
	  glBindBuffer( GL_ARRAY_BUFFER, out_vbo );
	  glBufferData( GL_ARRAY_BUFFER, 4 * 3 * in_F.rows() * sizeof(float), data, GL_STATIC_DRAW );
		
		free( data );
		
		return out_vbo;
	}
	
	GLuint computeVAO( const std::vector<GLuint>& in_vbos )
	{
	  GLuint out_vao = 0;
	  glGenVertexArrays( 1, &out_vao );
	  glBindVertexArray( out_vao );
		
		for( int i=0; i<in_vbos.size(); i++ )
		{
			glEnableVertexAttribArray(i);
	  	glBindBuffer( GL_ARRAY_BUFFER, in_vbos[i] );
	  	glVertexAttribPointer( i, 4, GL_FLOAT, GL_FALSE, 0, NULL );
		}
		
		return out_vao;
	}
	
	void finalizeVBO( GLuint& io_vbo )
	{
		glDeleteBuffers( 1, &io_vbo );
		io_vbo = 0;
	}
	
	void finalizeVBO( std::vector<GLuint>& io_vbos )
	{
		const int num_vbos = io_vbos.size();
		for( int i=0; i<num_vbos; i++ )
		{
			glDeleteBuffers( 1, &io_vbos[i] );
			io_vbos[i] = 0;
		}
		io_vbos.clear();
	}
	
	void finalizeVAO( GLuint& io_vao )
	{
		glDeleteVertexArrays( 1, &io_vao );
		io_vao = 0;
	}
	
	void drawVAO( GLuint& io_vao, int in_numTris )
	{
    glBindVertexArray( io_vao );
    glDrawArrays( GL_TRIANGLES, 0, in_numTris * 3 );		
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
	
	GLFWwindow* window()
	{
		return m_Window;
	}
	
	GLuint shader()
	{
		return m_Shader;
	}
	
private:
	int m_Width;
	int m_Height;
	GLFWwindow* m_Window;
	GLuint m_Shader;
	GLuint m_Texture;
	
	Image<float, 4> m_Buffer;
	unsigned char* m_BufferBytes;
	
	std::set<GLuint> m_GeneratedTextures;
};

void readMatrix( Eigen::Matrix4f& out_mat, boost::json::value& in_json, const std::string& in_outer_tag, const std::string& in_inner_tag )
{
	auto mat = boost::json::value_to<std::vector<std::vector<float>>>( in_json.as_object()[ in_outer_tag ].as_object()[ in_inner_tag ] );
	for( int j=0; j<mat.size(); j++ )
	{
		for( int i=0; i<mat[j].size(); i++ )
		{
			out_mat( j, i ) = mat[j][i];
		}
	}
}

//https://stackoverflow.com/questions/63429179/eigen-and-glm-products-produce-different-results
glm::mat4 EigenToGlmMat( const Eigen::Matrix4f& v )
{
    glm::mat4 result;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result[i][j] = v(j, i);
        }
    }
    return result;
}

glm::mat4 loadMVPFromJson( const std::string& in_json_filename, Eigen::Matrix4f& out_Eigen_mvp, Eigen::Matrix3f& out_Eigen_mv )
{
	std::ifstream json_file( in_json_filename );
	std::string content( std::istreambuf_iterator<char>{ json_file }, std::istreambuf_iterator<char>{} );
	boost::json::value camera_json = boost::json::parse( content );

	Eigen::Matrix4f model, view, projection ; 
	readMatrix( view, camera_json, "camera", "world" );
	view = view.inverse();
	
	readMatrix( projection, camera_json, "camera", "project_mat" );
	model = Eigen::Matrix4f::Identity();
	
	out_Eigen_mvp = projection * view * model;
	out_Eigen_mv = ( view * model ).block<3, 3>( 0, 0 );
	return EigenToGlmMat( projection ) * EigenToGlmMat( view ) * EigenToGlmMat( model );
}

template<typename T, int N>
void saveImageAsHDF5( const std::string& in_filename, const Image<T, N>& in_image, const int in_channel_idx )
{
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data;
	data.resize( in_image.getHeight(), in_image.getWidth() );
	
	for( int j=0; j<in_image.getHeight(); j++ )
	{
		for( int i=0; i<in_image.getWidth(); i++ )
		{
			data( j, i ) = in_image.getValue( i, j, in_channel_idx );
		}
	}
	
	HDF5File h5( in_filename, HDF5AccessType::READ_WRITE );
	Eigen::Vector2i resolution = Eigen::Vector2i{ in_image.getWidth(), in_image.getHeight() };
	h5.writeVector( "", "resolution", resolution );
	h5.writeMatrix( "", "data", data );
}

int main( int argc, char* argv[] )
{
	boost::program_options::options_description opt( "Option" );
	opt.add_options()
		("help", "help")
	  ("obj_file_template", boost::program_options::value<std::string>()->default_value("blender/object/object_%03d.obj"), "obj_file_template")
		("camera_file_template", boost::program_options::value<std::string>()->default_value("blender/camera_pm/camera_%03d.json"), "camera_file_template")
		("out_feature_dir", boost::program_options::value<std::string>()->default_value("f_cs"), "out_feature_dir")
		("width,w", boost::program_options::value<int>()->default_value(512), "width")
		("height,h", boost::program_options::value<int>()->default_value(512), "height")
		("frame_start", boost::program_options::value<int>()->default_value(1), "frame_start" )
		("frame_end", boost::program_options::value<int>()->default_value(1), "frame_end" )
		("frame_skip", boost::program_options::value<int>()->default_value(1), "frame_skip" );
			
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
	
	const std::string obj_file_template = vm["obj_file_template"].as<std::string>();
	std::cout << "obj_file_template: " << obj_file_template << std::endl;
	const std::string camera_file_template = vm["camera_file_template"].as<std::string>();
	std::cout << "camera_file_template: " << camera_file_template << std::endl;
	
	const std::string out_feature_dir = vm["out_feature_dir"].as<std::string>();
	std::cout << "out_feature_dir: " << out_feature_dir << std::endl;
	
	const int frame_start = vm["frame_start"].as<int>();
	const int frame_end = vm["frame_end"].as<int>();
	const int frame_skip = vm["frame_skip"].as<int>();
	std::cout << "frame start: " << frame_start << ", end: " << frame_end << ", skip: " << frame_skip << std::endl;
	
	std::filesystem::path out_feature_path = out_feature_dir;
	if( !std::filesystem::is_directory( out_feature_path ) ) std::filesystem::create_directory( out_feature_path );
	
	constexpr int buf_size = 4096;
	char buf[buf_size];
	
	SurfaceRenderer* renderer = new SurfaceRenderer( width, height );
	
	if( !renderer->initializeGL() )
		return -1;
	
	FrameBufferObject* fbo = new FrameBufferObject( width, height );
	
	Image<float, 1> v_rel_norm; v_rel_norm.init( width, height );
	Image<float, 4> image;
	image.init( width, height );
	
	std::chrono::steady_clock::time_point _begin, _end;
	
	for( int frame_idx = frame_start; frame_idx <= frame_end; frame_idx += frame_skip )
	{
		std::cout << "processing frame " << frame_idx << std::endl;
		_begin = std::chrono::steady_clock::now();
		
		// ### set up files
		snprintf( buf, buf_size, obj_file_template.c_str(), frame_idx );
		std::string obj_filename = buf;
				
		snprintf( buf, buf_size, obj_file_template.c_str(), frame_idx+1 );
		std::string next_obj_filename = buf;
		
		snprintf( buf, buf_size, camera_file_template.c_str(), frame_idx );
		std::string camera_filename = buf;
		
		snprintf( buf, buf_size, camera_file_template.c_str(), frame_idx+1 );
		std::string next_camera_filename = buf;
			
		std::string frame_d = std::to_string( frame_idx );
		std::filesystem::path frame_path = out_feature_dir; frame_path /= frame_d;
		if( !std::filesystem::is_directory( frame_path ) ) std::filesystem::create_directory( frame_path );
	
		std::filesystem::path normal_0_path = out_feature_dir; normal_0_path /= frame_d; normal_0_path /= "2d_apparent_normal_cam_0.hdf5";
		std::filesystem::path normal_1_path = out_feature_dir; normal_1_path /= frame_d; normal_1_path /= "2d_apparent_normal_cam_1.hdf5";
		std::filesystem::path normal_2_path = out_feature_dir; normal_2_path /= frame_d; normal_2_path /= "2d_apparent_normal_cam_2.hdf5";
		
		std::filesystem::path vel_0_path = out_feature_dir; vel_0_path /= frame_d; vel_0_path /= "2d_velocity_cam_0.hdf5";
		std::filesystem::path vel_1_path = out_feature_dir; vel_1_path /= frame_d; vel_1_path /= "2d_velocity_cam_1.hdf5";
		std::filesystem::path vel_norm_path = out_feature_dir; vel_norm_path /= frame_d; vel_norm_path /= "2d_velocity_cam_norm.hdf5";
		
		std::filesystem::path gaussian_path = out_feature_dir; gaussian_path /= frame_d; gaussian_path /= "2d_gaussian_curvature.hdf5";
		std::filesystem::path mean_path = out_feature_dir; mean_path /= frame_d; mean_path /= "2d_mean_curvature.hdf5";
				
		Eigen::MatrixXf V;
		Eigen::MatrixXi F;
		Eigen::MatrixXf N;
		Eigen::MatrixXf PD1, PD2, PV1, PV2;
			
		igl::read_triangle_mesh( obj_filename, V, F );
			
		igl::per_vertex_normals( V, F, N );
		igl::principal_curvature( V, F, PD1, PD2, PV1, PV2 );
		Eigen::MatrixXf GM; GM.resize( V.rows(), 2 );
		for( int i=0; i<GM.rows(); i++ )
		{
			GM.col(0)(i) = PV1.col(0)(i) * PV2.col(0)(i);
			GM.col(1)(i) = ( PV1.col(0)(i) + PV2.col(0)(i) ) * 0.5;
		}
			
		Eigen::Matrix4f eigen_mvp;
		Eigen::Matrix3f eigen_mv;
		glm::mat4 mat = loadMVPFromJson( camera_filename, eigen_mvp, eigen_mv );
	
		for( int i=0; i<N.rows(); i++ )
		{
			Eigen::Vector3f _n = N.row(i);
			N.row(i) = eigen_mv * _n;
		}
		
		float fps = 24.0;
		Eigen::MatrixXf V_n;
		Eigen::MatrixXi F_n;
		igl::read_triangle_mesh( next_obj_filename, V_n, F_n );
		
		Eigen::Matrix4f eigen_mvp_n;
		Eigen::Matrix3f eigen_mv_n;
		glm::mat4 mat_n = loadMVPFromJson( next_camera_filename, eigen_mvp_n, eigen_mv_n );
	
		Eigen::MatrixXf sc; sc.resize( V.rows(), 4 );
		for( int i=0; i<V.rows(); i++ )
		{
			Eigen::Vector4f v; v << V.row(i)(0), V.row(i)(1), V.row(i)(2), 1.0;
			sc.row(i) = eigen_mvp * v;
			sc.row(i) /= sc.row(i)(3);
			sc.row(i)(0) *= 0.5;
			sc.row(i)(1) *= 0.5;
		}
	
		Eigen::MatrixXf sc_n; sc_n.resize( V.rows(), 4 );
		for( int i=0; i<V.rows(); i++ )
		{
			Eigen::Vector4f v_n; v_n << V_n.row(i)(0), V_n.row(i)(1), V_n.row(i)(2), 1.0;
			sc_n.row(i) = eigen_mvp_n * v_n;
			sc_n.row(i) /= sc_n.row(i)(3);
			sc_n.row(i)(0) *= 0.5;
			sc_n.row(i)(1) *= 0.5;
		}
	
		Eigen::MatrixXf rel_v;
		rel_v.resize( V.rows(), 2 );
		for( int i=0; i<V.rows(); i++ )
		{
			Eigen::Vector4f v_sc_n = sc_n.row(i);
			Eigen::Vector4f v_sc = sc.row(i);
			rel_v.row(i) = ( v_sc_n.segment<2>(0) - v_sc.segment<2>(0) ) * fps;
		}
				
		GLuint vbo_pos = renderer->computeVBO( V, F );
		GLuint vbo_normal = renderer->computeVBO( N, F );
		GLuint vbo_curvatures = renderer->computeVBO( GM, F );
		GLuint vbo_relative_velocity = renderer->computeVBO( rel_v, F );

		/* normal */
		std::vector<GLuint> vbos_normal;
		vbos_normal.push_back( vbo_pos );
		vbos_normal.push_back( vbo_normal );	
		GLuint vao_normal = renderer->computeVAO( vbos_normal );
		
		fbo->bind();
		renderer->clear();
		
		glEnable( GL_DEPTH_TEST );
		glUseProgram( renderer->shader() );
		renderer->setMatrix( renderer->shader(), mat );
		renderer->drawVAO( vao_normal, F.rows() );
		glDisable( GL_DEPTH_TEST );
				
		renderer->readBuffer( image );
		saveImageAsHDF5( normal_0_path.string(), image, 0 );
		saveImageAsHDF5( normal_1_path.string(), image, 1 );
		saveImageAsHDF5( normal_2_path.string(), image, 2 );
		
		fbo->unbind();
		
		/* curvature */
		std::vector<GLuint> vbos_curvature;
		vbos_curvature.push_back( vbo_pos );
		vbos_curvature.push_back( vbo_curvatures );	
		GLuint vao_curvature = renderer->computeVAO( vbos_curvature );
		
		fbo->bind();
		renderer->clear();
		
		glEnable( GL_DEPTH_TEST );
		glUseProgram( renderer->shader() );
		renderer->setMatrix( renderer->shader(), mat );
		renderer->drawVAO( vao_curvature, F.rows() );
		glDisable( GL_DEPTH_TEST );
		
		renderer->readBuffer( image );
		saveImageAsHDF5( gaussian_path.string(), image, 0 );
		saveImageAsHDF5( mean_path.string(), image, 1 );
		
		fbo->unbind();
		
		/* relative velocity */
		std::vector<GLuint> vbos_rel_v;
		vbos_rel_v.push_back( vbo_pos );
		vbos_rel_v.push_back( vbo_relative_velocity );	
		GLuint vao_rel_v = renderer->computeVAO( vbos_rel_v );
		
		fbo->bind();
		renderer->clear();
		
		glEnable( GL_DEPTH_TEST );
		glUseProgram( renderer->shader() );
		renderer->setMatrix( renderer->shader(), mat );
		renderer->drawVAO( vao_rel_v, F.rows() );
		glDisable( GL_DEPTH_TEST );
		
		renderer->readBuffer( image );
		saveImageAsHDF5( vel_0_path.string(), image, 0 );
		saveImageAsHDF5( vel_1_path.string(), image, 1 );
		
		for( int i=0; i<width*height; i++ )
			v_rel_norm.getPtr()[i] = std::sqrt( image.getPtr()[i*4] * image.getPtr()[i*4] + image.getPtr()[i*4+1] * image.getPtr()[i*4+1] );
		
		saveImageAsHDF5( vel_norm_path.string(), v_rel_norm, 0 );
		
		fbo->unbind();
		
		renderer->finalizeVBO( vbo_pos );
		renderer->finalizeVBO( vbo_normal );
		renderer->finalizeVBO( vbo_curvatures );
		renderer->finalizeVBO( vbo_relative_velocity );
		renderer->finalizeVAO( vao_normal );
		renderer->finalizeVAO( vao_curvature );
		renderer->finalizeVAO( vao_rel_v );
		
		_end = std::chrono::steady_clock::now();
		const float elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _begin).count() / 1000000.0;
		std::cout << "Elapsed time: " << elapsed_time << " [s]" << std::endl;
	}
	delete fbo;
	delete renderer;
		
	return 0;
}
