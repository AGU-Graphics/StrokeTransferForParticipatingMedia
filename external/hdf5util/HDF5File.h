// This file was originally created by Breannan Smith and has been used for SCISIM, Continuum Foam, and Hybrid Grain projects.
// The original file has been slightly modified (by Yonghao Yue) for handling data types necessary for the Stroke Transfer for Participating Media project.

#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "hdf5.h"
#include <iostream>

template <herr_t H5CloseOperation( hid_t id )>
class HDFID final
{
public:
  HDFID()
  : m_hid_t( -1 )
  {}

  explicit HDFID( const hid_t value )
  : m_hid_t( value )
  {}

  ~HDFID()
  {
    if( m_hid_t >= 0 )
    {
      H5CloseOperation( m_hid_t );
    }
  }

  HDFID( HDFID&& other )
  : m_hid_t( other.m_hid_t )
  {
    other.m_hid_t = -1;
  }

  HDFID& operator=( HDFID&& other )
  {
    const hid_t others_hid_t{ other.m_hid_t };
    other.m_hid_t = -1;
    m_hid_t = others_hid_t;
    return *this;
  }

  operator hid_t() const
  {
    return m_hid_t;
  }

private:

  HDFID( const HDFID& ) = delete;
  HDFID& operator=( const HDFID& ) = delete;

  hid_t m_hid_t;

};

namespace HDF5SupportedTypes
{
  template <typename T>
  constexpr bool isSupportedEigenType()
  {
    return false;
  }

  template <>
  constexpr bool isSupportedEigenType<int>()
  {
    return true;
  }
	
  template <>
  constexpr bool isSupportedEigenType<int64_t>()
  {
    return true;
  }

  template<>
  constexpr bool isSupportedEigenType<long>()
  {
    return true;
  }

  template <>
  constexpr bool isSupportedEigenType<unsigned>()
  {
    return true;
  }

  template <>
  constexpr bool isSupportedEigenType<unsigned long>()
  {
    return true;
  }

  template <>
  constexpr bool isSupportedEigenType<float>()
  {
    return true;
  }

  template <>
  constexpr bool isSupportedEigenType<double>()
  {
    return true;
  }
}

enum class HDF5AccessType : std::uint8_t
{
  READ_ONLY,
  READ_WRITE
};

class HDF5File final
{

public:

  HDF5File();
  HDF5File( const std::string& file_name, const HDF5AccessType& access_type );
  ~HDF5File();

  hid_t fileID();

  void open( const std::string& file_name, const HDF5AccessType& access_type );

  bool is_open() const;

  HDFID<H5Gclose> getGroup( const std::string& group_name ) const;

  HDFID<H5Gclose> findGroup( const std::string& group_name ) const;

  void writeString( const std::string& group, const std::string& variable_name, const std::string& string_variable ) const;

  void readString( const std::string& group, const std::string& variable_name, std::string& string_variable ) const;

  void writedouble( const std::string& group, const std::string& variable_name, const double& variable ) const
  {
    Eigen::Matrix<double,1,1> output_mat;
    output_mat << variable;
    writeMatrix( group, variable_name, output_mat );
  }

  void readdouble( const std::string& group, const std::string& variable_name, double& variable ) const
  {
    Eigen::Matrix<double,1,1> input_mat;
    readMatrix( group, variable_name, input_mat );
    variable = input_mat( 0, 0 );
  }

  void writeint( const std::string& group, const std::string& variable_name, const int& variable ) const
  {
    Eigen::Matrix<int,1,1> output_mat;
    output_mat << variable;
    writeMatrix( group, variable_name, output_mat );
  }

  void readint( const std::string& group, const std::string& variable_name, int& variable ) const
  {
    Eigen::Matrix<int,1,1> input_mat;
    readMatrix( group, variable_name, input_mat );
    variable = input_mat( 0, 0 );
  }
	
	template <typename Derived>
	void writeVector( const std::string& group, const std::string& variable_name, const Eigen::DenseBase<Derived>& eigen_variable ) const
	{
    using HDFSID = HDFID<H5Sclose>;
    using HDFGID = HDFID<H5Gclose>;
    using HDFDID = HDFID<H5Dclose>;
		
    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );
		
		assert( eigen_variable.rows() >= 0 ); assert( eigen_variable.cols() >= 0 );
		if( !( eigen_variable.rows() == 1 || eigen_variable.cols() == 1 ) )
		{
			throw std::string{ "eigen_variable must be a row or column vector" };
		}
		
		const hsize_t dims[1] = { std::max<hsize_t>( hsize_t( eigen_variable.rows() ), hsize_t( eigen_variable.cols() ) ) };
		const HDFSID dataspace_id{ H5Screate_simple( 1, dims, nullptr ) };
		
    if( dataspace_id < 0 )
    {
      throw std::string{ "Failed to create HDF data space" };
    }
		
		const HDFGID grp_id{ getGroup( group ) };
		
    const HDFDID dataset_id{ H5Dcreate2( grp_id, variable_name.c_str(), computeHDFType( eigen_variable ), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) };
    if( dataset_id < 0 )
    {
      throw std::string{ "Failed to create HDF data set" };
    }		
		
    const herr_t status_write{ H5Dwrite( dataset_id, computeHDFType( eigen_variable ), H5S_ALL, H5S_ALL, H5P_DEFAULT, eigen_variable.derived().data() ) };
    if( status_write < 0 )
    {
      throw std::string{ "Failed to write HDF data" };
    }		
	}
	
	template <typename Derived>
	void readVector( const std::string& group, const std::string& variable_name, Eigen::DenseBase<Derived>& eigen_variable ) const
	{
    using HDFDID = HDFID<H5Dclose>;
    using HDFGID = HDFID<H5Gclose>;

    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );
		
    const HDFGID grp_id{ findGroup( group ) };

    const HDFDID dataset_id{ H5Dopen2( grp_id, variable_name.c_str(), H5P_DEFAULT ) };

    if( dataset_id < 0 )
    {
      throw std::string{ "Failed to open HDF data set" };
    }
    if( getNativeType( dataset_id ) != computeHDFType( eigen_variable ) )
    {
      throw std::string{ "Requested HDF data set is not of given type from Eigen variable" };
    }

    Eigen::ArrayXi dimensions;
    getDimensions( dataset_id, dimensions );
		
    if( dimensions.size() != 1 )
    {
      throw std::string{ "Invalid dimensions for Eigen vector type in file" };
    }
    if( ( dimensions < 0 ).any() )
    {
      throw std::string{ "Negative dimensions for Eigen vector type in file" };
    }
    
    eigen_variable.derived().resize( dimensions( 0 ) );

    const herr_t read_status{ H5Dread( dataset_id, getNativeType( dataset_id ), H5S_ALL, H5S_ALL, H5P_DEFAULT, eigen_variable.derived().data() ) };
    if( read_status < 0 )
    {
      throw std::string{ "Failed to read data from HDF file" };
    }
	}

  template <typename Derived>
  void writeMatrix( const std::string& group, const std::string& variable_name, const Eigen::DenseBase<Derived>& eigen_variable ) const
  {
    using HDFSID = HDFID<H5Sclose>;
    using HDFGID = HDFID<H5Gclose>;
    using HDFDID = HDFID<H5Dclose>;

    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );

    assert( eigen_variable.rows() >= 0 ); assert( eigen_variable.cols() >= 0 );
    const hsize_t dims[2] = { hsize_t( eigen_variable.rows() ), hsize_t( eigen_variable.cols() ) };
    const HDFSID dataspace_id{ H5Screate_simple( 2, dims, nullptr ) };
    if( dataspace_id < 0 )
    {
      throw std::string{ "Failed to create HDF data space" };
    }

    const HDFGID grp_id{ getGroup( group ) };

    const HDFDID dataset_id{ H5Dcreate2( grp_id, variable_name.c_str(), computeHDFType( eigen_variable ), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) };
    if( dataset_id < 0 )
    {
      throw std::string{ "Failed to create HDF data set" };
    }

    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> col_major_output_data;
    if( isColumnMajor( eigen_variable ) )
    {
      col_major_output_data.resize( eigen_variable.rows(), eigen_variable.cols() );
      col_major_output_data = eigen_variable.derived().matrix();
    }

    const herr_t status_write{ H5Dwrite( dataset_id, computeHDFType( eigen_variable ), H5S_ALL, H5S_ALL, H5P_DEFAULT, isColumnMajor( eigen_variable ) ? col_major_output_data.data() : eigen_variable.derived().data() ) };
    if( status_write < 0 )
    {
      throw std::string{ "Failed to write HDF data" };
    }
  }

  template <typename Derived>
  void readMatrix( const std::string& group, const std::string& variable_name, Eigen::DenseBase<Derived>& eigen_variable ) const
  {
    using HDFDID = HDFID<H5Dclose>;
    using HDFGID = HDFID<H5Gclose>;

    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );

    const HDFGID grp_id{ findGroup( group ) };

    const HDFDID dataset_id{ H5Dopen2( grp_id, variable_name.c_str(), H5P_DEFAULT ) };

    if( dataset_id < 0 )
    {
      throw std::string{ "Failed to open HDF data set" };
    }
    if( getNativeType( dataset_id ) != computeHDFType( eigen_variable ) )
    {
      throw std::string{ "Requested HDF data set is not of given type from Eigen variable" };
    }

    Eigen::ArrayXi dimensions;
    getDimensions( dataset_id, dimensions );
		
    if( dimensions.size() != 2 )
    {
      throw std::string{ "Invalid dimensions for Eigen matrix type in file" };
    }
    if( ( dimensions < 0 ).any() )
    {
      throw std::string{ "Negative dimensions for Eigen matrix type in file" };
    }

    if( rowsFixed( eigen_variable ) && eigen_variable.rows() != dimensions( 0 ) )
    {
      throw std::string{ "Eigen type of fixed row size does not have correct number of rows" };
    }
    if( colsFixed( eigen_variable ) && eigen_variable.cols() != dimensions( 1 ) )
    {
      throw std::string{ "Eigen type of fixed cols size does not have correct number of cols" };
    }

    eigen_variable.derived().resize( dimensions( 0 ), dimensions( 1 ) );

    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> col_major_input_data;
    if( isColumnMajor( eigen_variable ) )
    {
      col_major_input_data.resize( eigen_variable.rows(), eigen_variable.cols() );
    }

    const herr_t read_status{ H5Dread( dataset_id, getNativeType( dataset_id ), H5S_ALL, H5S_ALL, H5P_DEFAULT, isColumnMajor( eigen_variable ) ? col_major_input_data.data() : eigen_variable.derived().data() ) };
    if( read_status < 0 )
    {
      throw std::string{ "Failed to read data from HDF file" };
    }
    if( isColumnMajor( eigen_variable ) )
    {
      eigen_variable.derived().matrix() = col_major_input_data;
    }
  }
	
  template <typename Derived>
  void write1DArrayAsTensor( const std::string& group, const std::string& variable_name, const Eigen::DenseBase<Derived>& eigen_variable, const Eigen::ArrayXi& dimensions ) const
  {
    using HDFSID = HDFID<H5Sclose>;
    using HDFGID = HDFID<H5Gclose>;
    using HDFDID = HDFID<H5Dclose>;

    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );

    assert( eigen_variable.rows() >= 0 ); assert( eigen_variable.cols() >= 0 );
		
		int num_elems = 1;		
		hsize_t *dims = new hsize_t[ dimensions.size() ];
		for( int i=0; i<dimensions.size(); i++ )
		{
			dims[i] = dimensions(i);
			num_elems *= dims[i];
		}
		
    const HDFSID dataspace_id{ H5Screate_simple( dimensions.size(), dims, nullptr ) };
		delete[] dims;
    if( dataspace_id < 0 )
    {
      throw std::string{ "Failed to create HDF data space" };
    }
		
		if( num_elems != eigen_variable.size() )
		{
			throw std::string{ "Unmatched size" };
		}

    const HDFGID grp_id{ getGroup( group ) };

    const HDFDID dataset_id{ H5Dcreate2( grp_id, variable_name.c_str(), computeHDFType( eigen_variable ), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) };
    if( dataset_id < 0 )
    {
      throw std::string{ "Failed to create HDF data set" };
    }

    const herr_t status_write{ H5Dwrite( dataset_id, computeHDFType( eigen_variable ), H5S_ALL, H5S_ALL, H5P_DEFAULT, eigen_variable.derived().data() ) };
    if( status_write < 0 )
    {
      throw std::string{ "Failed to write HDF data" };
    }
  }
	
  template <typename Derived>
  void readTensorAs1DArray( const std::string& group, const std::string& variable_name, Eigen::MatrixBase<Derived>& array, Eigen::ArrayXi& dimensions ) const
  {
    using HDFDID = HDFID<H5Dclose>;
    using HDFGID = HDFID<H5Gclose>;

    using Scalar = typename Derived::Scalar;
    static_assert( HDF5SupportedTypes::isSupportedEigenType<Scalar>(), "Error, scalar type of Eigen variable must be float, double, unsigned or integer" );

    const HDFGID grp_id{ findGroup( group ) };

    const HDFDID dataset_id{ H5Dopen2( grp_id, variable_name.c_str(), H5P_DEFAULT ) };
		
		if( dataset_id < 0 )
    {
      throw std::string{ "Failed to open HDF data set" };
    }

		getDimensions( dataset_id, dimensions );
		
		if( ( dimensions < 0 ).any() )
    {
      throw std::string{ "Negative dimensions for Eigen matrix type in file" };
    }
    
		int num_elements = 1;
		for( int i=0; i<dimensions.size(); i++ )
			num_elements *= dimensions( i );
		
    array.derived().resize( num_elements );

    const herr_t read_status{ H5Dread( dataset_id, computeHDFType( array ), H5S_ALL, H5S_ALL, H5P_DEFAULT, array.derived().data() ) };
    if( read_status < 0 )
    {
      throw std::string{ "Failed to read data from HDF file" };
    }
  }

private:

  static hid_t getNativeType( const hid_t dataset_id )
  {
    using HDFTID = HDFID<H5Tclose>;

    const HDFTID dtype_id{ H5Dget_type( dataset_id ) };
    if( dtype_id < 0 )
    {
      throw std::string{ "Failed to get HDF5 datatype from dataset" };
    }
    if( H5Tequal( dtype_id, H5T_NATIVE_DOUBLE ) > 0 )
    {
      return H5T_NATIVE_DOUBLE;
    }
    else if( H5Tequal( dtype_id, H5T_NATIVE_FLOAT ) > 0 )
    {
      return H5T_NATIVE_FLOAT;
    }
    else if( H5Tequal( dtype_id, H5T_NATIVE_INT ) > 0 )
    {
      return H5T_NATIVE_INT;
    }
    else if( H5Tequal( dtype_id, H5T_NATIVE_INT64 ) > 0 )
    {
      return H5T_NATIVE_INT64;
    }
    else if( H5Tequal( dtype_id, H5T_NATIVE_UINT ) > 0 )
    {
      return H5T_NATIVE_UINT;
    }
    else return -1;
  }

  static void getDimensions( const hid_t dataset_id, Eigen::ArrayXi& dimensions )
  {
    using HDFSID = HDFID<H5Sclose>;

    const HDFSID space_id{ H5Dget_space( dataset_id ) };
    if( space_id < 0 )
    {
      throw std::string{ "Failed to open data space" };
    }
    const int rank{ H5Sget_simple_extent_ndims( space_id ) };
    if( rank < 0 )
    {
      throw std::string{ "Failed to get rank" };
    }
    dimensions.resize( rank );
    std::vector<hsize_t> dims( static_cast<std::vector<hsize_t>::size_type>( rank ) );
    assert( int( dims.size() ) == rank );
    const herr_t status_get_simple_extent_dims{ H5Sget_simple_extent_dims( space_id, dims.data(), nullptr ) };
    if( status_get_simple_extent_dims < 0 )
    {
      throw std::string{ "Failed to get extents" };
    }
    for( int i = 0; i < rank; ++i )
    {
      dimensions( i ) = int( dims[i] );
    }
  }

  template <typename ScalarType>
  static constexpr hid_t computeHDFType()
  {
    using std::is_same;
    return is_same<ScalarType,double>::value ? H5T_NATIVE_DOUBLE : is_same<ScalarType,float>::value ? H5T_NATIVE_FLOAT : is_same<ScalarType,int64_t>::value ? H5T_NATIVE_INT64 : is_same<ScalarType,int>::value ? H5T_NATIVE_INT : is_same<ScalarType,unsigned>::value ? H5T_NATIVE_UINT : is_same<ScalarType,unsigned long>::value ? H5T_NATIVE_ULONG : is_same<ScalarType,long>::value ? H5T_NATIVE_LONG : -1;
  }

  template <typename Derived>
  static constexpr hid_t computeHDFType( const Eigen::EigenBase<Derived>& )
  {
    return computeHDFType<typename Derived::Scalar>();
  }

  template <typename Derived>
  static constexpr bool isColumnMajor( const Eigen::EigenBase<Derived>& )
  {
    return !Derived::IsRowMajor;
  }

  template <typename Derived>
  static constexpr bool rowsFixed( const Eigen::EigenBase<Derived>& )
  {
    return Derived::RowsAtCompileTime != Eigen::Dynamic;
  }

  template <typename Derived>
  static constexpr bool colsFixed( const Eigen::EigenBase<Derived>& )
  {
    return Derived::ColsAtCompileTime != Eigen::Dynamic;
  }

  hid_t m_hdf_file_id;
  bool m_file_opened;

};

#endif
