// This file was originally created by Breannan Smith and has been used for SCISIM, Continuum Foam, and Hybrid Grain projects.
// The original file has been slightly modified (by Yonghao Yue) for handling data types necessary for the Stroke Transfer for Participating Media project.

#include "Utils.h"
#include <stdio.h>
#include <vector>

void splitAtLastCharacterOccurence( const std::string& input_string, std::string& left_substring, std::string& right_substring, const char chr )
{
  const std::string::size_type position = input_string.find_last_of( chr );
  left_substring = input_string.substr( 0, position );
  if( position != std::string::npos )
  {
    right_substring = input_string.substr( position + 1 );
  }
  else
  {
    right_substring = "";
  }
}

std::string removeWhiteSpace( const std::string& input_string )
{
  std::string output_string = input_string;
  output_string.erase( std::remove_if( output_string.begin(), output_string.end(), []( const char x ){ return std::isspace( x ); } ), output_string.end() );
  return output_string;
}

std::string trimCharacterRight( const std::string& input_string, const char chr )
{
  return input_string.substr( 0, input_string.find_last_not_of( chr ) + 1 );
}

void tokenize( const std::string& str, const char chr, std::vector<std::string>& tokens )
{
  std::string::size_type substring_start = 0;
  std::string::size_type substring_end = str.find_first_of( chr, substring_start );
  while( substring_end != std::string::npos )
  {
    tokens.emplace_back( str.substr( substring_start, substring_end - substring_start ) );
    substring_start = substring_end + 1;
    substring_end = str.find_first_of( chr, substring_start );
  }

  if( substring_start < str.size() )
  {
    tokens.emplace_back( str.substr( substring_start ) );
  }

  if( str.back() == chr )
  {
    tokens.emplace_back( "" );
  }
}

std::vector<std::string> tokenize( const std::string& str, const char delimiter )
{
  std::vector<std::string> tokens;
  tokenize( str, delimiter, tokens );
  return tokens;
}

bool readDoubleList( const std::string& input_text, const char delimiter, Eigen::Vector3d& list )
{
  const std::vector<std::string> split_input = tokenize( input_text, delimiter );
  for( unsigned entry_number = 0; entry_number < 3; ++entry_number )
  {
    if( !extractFromString( split_input[entry_number], list(entry_number) ) )
    {
      return false;
    }
  }
  return true;
}

bool readDoubleList( const std::string& input_text, const char delimiter, Eigen::Vector4d& list )
{
	const std::vector<std::string> split_input = tokenize( input_text, delimiter );
	for( unsigned entry_number = 0; entry_number < 4; ++entry_number )
	{
		if( !extractFromString( split_input[entry_number], list(entry_number) ) )
		{
			return false;
		}
	}
	return true;
}

std::string::size_type computeNumCharactersToRight( const std::string& input_string, const char chr )
{
  const std::string::size_type position = input_string.find_first_of( chr );
  if( position == std::string::npos )
  {
    return 0;
  }
  return input_string.length() - input_string.find_first_of( chr ) - 1;
}

unsigned computeNumDigits( unsigned n )
{
  if( n == 0 ) { return 1; }
  unsigned num_digits{ 0 };
  while( n != 0 )
  {
    n /= 10;
    ++num_digits;
  }
  return num_digits;
}
