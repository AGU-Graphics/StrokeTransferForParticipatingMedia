// This file was originally created by Breannan Smith and has been used for SCISIM, Continuum Foam, and Hybrid Grain projects.
// The original file has been slightly modified (by Yonghao Yue) for handling data types necessary for the Stroke Transfer for Participating Media project.

#ifndef Utils_h
#define Utils_h

#include <string>
#include <vector>
#include <sstream>
#include <Eigen/Dense>

template<typename T> inline T PI()
{
  return T(3.14159265358979323846264338327950288419716939937510582097494459230);
}

void splitAtLastCharacterOccurence( const std::string& input_string, std::string& left_substring, std::string& right_substring, const char chr );
std::string removeWhiteSpace( const std::string& input_string );
std::string trimCharacterRight( const std::string& input_string, const char chr );
void tokenize( const std::string& str, const char chr, std::vector<std::string>& tokens );
std::vector<std::string> tokenize( const std::string& str, const char delimiter );
std::string::size_type computeNumCharactersToRight( const std::string& input_string, const char chr );
unsigned computeNumDigits( unsigned n );

template<class T>
bool extractFromString( const std::string& str, T& res )
{
  std::stringstream input_strm( str );
  input_strm >> res;
  return !input_strm.fail();
}

template<class T, int N>
bool extractFromString( const std::string& str, Eigen::Matrix<T, N, 1>& vec )
{
  std::stringstream input_strm( str );
  for( int i=0; i<vec.size(); i++ )
    input_strm >> vec(i);
  return !input_strm.fail();
}

bool readDoubleList( const std::string& input_text, const char delimiter, Eigen::Vector3d& list );
bool readDoubleList( const std::string& input_text, const char delimiter, Eigen::Vector4d& list );

#endif /* Utils_h */
