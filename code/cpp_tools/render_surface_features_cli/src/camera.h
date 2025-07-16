// -----------------------------------------------------------------------------
// Stroke Transfer for Participating Media
// http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
//
// File: render_surface_features_cli/src/camera.h
// Maintainer: Yonghao Yue
//
// Description:
// This file implements a simple camera model.
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


#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <stdint.h>
#include <Eigen/Dense>
#include <lib/image/image.h>

const double _FAR_ = 1.0e-38;

template<typename T>
struct InitialCameraSettings
{
  Eigen::Matrix<T, 3, 1> eyePoint;
  Eigen::Matrix<T, 3, 1> lookAtLocation;
  Eigen::Matrix<T, 3, 1> upVector; //not necessarily perpendicular to the internal look-at vector.
  T irisF;
  T focalLength;
};

template<typename T>
struct CameraInternal
{
  Eigen::Matrix<T, 3, 1> zVector;

  T focalLength;
  T focusLength;
  T t_i;

  int32_t width;
  int32_t height;
  
  T irisF;
};

template<typename T>
struct Camera
{
  Eigen::Matrix<T, 3, 1> eyePoint;
  Eigen::Matrix<T, 3, 1> xVector;
  Eigen::Matrix<T, 3, 1> yVector;
  Eigen::Matrix<T, 3, 1> planeCenter;

  T screenWidth;
  T screenHeight;

  T invWidth;
  T invHeight;

  CameraInternal<T> internalData;
};

template<typename T> T BBPI() { return T(3.14159265358979323846264338327950288); }

template<typename T>
inline void rotateVector(Eigen::Matrix<T, 3, 1>* out_v, const Eigen::Matrix<T, 3, 1>& in_v, const Eigen::Matrix<T, 3, 1>& in_axis, const T in_angle_rad)
{
  Eigen::Matrix<T, 3, 1> axis { in_axis };
  axis.normalize();
  
  const T cos_a = cos(in_angle_rad*T(0.5));
  const T sin_a = sin(in_angle_rad*T(0.5));
  const T m_sin_a = -sin_a;
  
  //(r * p * q).v
  //p = (0, px, py, pz)
  //q = (cos_a, axis_x * sin_a, axis_y * sin_a, axis_z * sin_a)
  //r = (cos_a, axis_x * m_sin_a, axis_y * m_sin_a, axis_z * m_sin_a)

  const T r_p_t = -m_sin_a * axis.dot(in_v);
  const Eigen::Matrix<T, 3, 1> r_p = cos_a * in_v + m_sin_a * in_v.cross(axis);
  *out_v = sin_a * r_p_t * axis + cos_a * r_p + sin_a * axis.cross(r_p);
}

template<typename T, int N>
bool checkResolutionConsistency(const Camera<T>& in_Camera, const Image<T,N>& in_image )
{
  return in_Camera.internalData.width == in_image.width && in_Camera.internalData.height == in_image.height;
}

template<typename T>
void setPlane(Camera<T>& io_Camera)
{
  io_Camera.planeCenter = io_Camera.internalData.t_i * io_Camera.internalData.zVector;
}

template<typename T>
void recalcLensData(Camera<T>& io_Camera)
{
  io_Camera.internalData.t_i = io_Camera.internalData.focalLength;
  setPlane<T>(io_Camera);
}

template<typename T>
void setEyePoint(const Eigen::Matrix<T, 3, 1>& in_EyePoint, Camera<T>& io_Camera)
{
  io_Camera.eyePoint = in_EyePoint;
}

template<typename T>
void setFocalLength(const T in_FocalLength, Camera<T>& io_Camera)
{
  io_Camera.internalData.focalLength = in_FocalLength;
  recalcLensData<T>(io_Camera);
}

template<typename T>
void setIrisF(const T in_IrisF, Camera<T>& io_Camera)
{
  io_Camera.internalData.irisF = in_IrisF;
  recalcLensData<T>(io_Camera);
}

template<typename T>
void lookAt(const Eigen::Matrix<T, 3, 1>& in_LookAt, const Eigen::Matrix<T, 3, 1>& in_Up, Camera<T>& io_Camera)
{
  const Eigen::Matrix<T, 3, 1> v { in_LookAt - io_Camera.eyePoint };
  const T len = v.norm();

  io_Camera.internalData.focusLength = len;
  io_Camera.internalData.zVector = -v / len;
  
  const T dot_up_z = in_Up.dot(io_Camera.internalData.zVector);
  Eigen::Matrix<T, 3, 1> _y { in_Up - dot_up_z * io_Camera.internalData.zVector };
  _y.normalize();
  io_Camera.yVector = _y;

  io_Camera.xVector = io_Camera.yVector.cross( io_Camera.internalData.zVector );
  recalcLensData<T>(io_Camera);
}

template<typename T>
void setViewportSize(const int32_t in_Width, const int32_t in_Height, Camera<T>& io_Camera)
{
  io_Camera.internalData.width = in_Width;
  io_Camera.invWidth = T(1.0) / in_Width;
  io_Camera.internalData.height = in_Height;
  io_Camera.invHeight = T(1.0) / in_Height;

  //sw <= 0.036, sh <= 0.024
  if(in_Width >= T(1.5)*in_Height)
  {
    io_Camera.screenWidth = T(0.036);
    io_Camera.screenHeight = in_Height * T(0.036) * io_Camera.invWidth;
  }
  else
  {
    io_Camera.screenWidth = in_Width * T(0.024) * io_Camera.invHeight;
    io_Camera.screenHeight = T(0.024);
  }
}

template<typename T>
void prepareCameraWithCameraSettings(const int32_t in_Width, const int32_t in_Height, const InitialCameraSettings<T>& in_Settings, Camera<T>& io_Camera)
{
  io_Camera.internalData.focalLength = 100.0;
  io_Camera.internalData.focusLength = 1.0;
  io_Camera.internalData.irisF = 16.0;

  setEyePoint<T>(in_Settings.eyePoint, io_Camera);
  lookAt<T>(in_Settings.lookAtLocation, in_Settings.upVector, io_Camera);
  setIrisF(in_Settings.irisF, io_Camera);
  setFocalLength(in_Settings.focalLength, io_Camera);
  setViewportSize(in_Width, in_Height, io_Camera);
}

template<typename T>
void getLookAt(Eigen::Matrix<T, 3, 1>& out_V, const Camera<T>& in_Camera)
{
  out_V = in_Camera.eyePoint - in_Camera.internalData.zVector * in_Camera.internalData.focusLength;
}

template<typename T>
void moveUpCameraCoord(const T in_Dist, Camera<T>& io_Camera)
{
  io_Camera->eyePoint += in_Dist * io_Camera.yVector;
}

template<typename T>
void moveUpGlobalCoord(const T in_Dist, Camera<T>& io_Camera)
{
  io_Camera->eyePoint(1) += in_Dist;
}

template<typename T>
void moveFrontFixUp(const T in_Dist, Camera<T>& io_Camera)
{
  Eigen::Matrix<T, 3, 1> d { io_Camera.internalData.zVector(0), T(0.0), io_Camera.internalData.zVector(2) };
  if( d.norm() < T(0.0000001))
    return;
  d.normalize();
  io_Camera.eyePoint -= d * in_Dist;
}

template<typename T>
void moveFrontFixUpFixLookAt(const T in_Dist, Camera<T>& io_Camera)
{
  Eigen::Matrix<T, 3, 1> d { io_Camera.internalData.zVector(0), T(0.0), io_Camera.internalData.zVector(2) };
  if( d.norm() < T(0.0000001))
    return;
  d.normalize();

  const T newFocusLength = max(io_Camera.internalData.focalLength, io_Camera.internalData.focusLength-in_Dist);
  const T the_Dist = io_Camera.internalData.focusLength - newFocusLength;

  io_Camera.eyePoint -= d * the_Dist;
  io_Camera.internalData.focusLength = newFocusLength;
}

template<typename T>
void rotateFixEyePointCameraCoord(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  //rotate around y-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.yVector, in_HorizontalAngle);

  //rotate around x-axis
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.xVector, in_VerticalAngle);

  //rotate around z-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.internalData.zVector, -in_RoundAngle);

  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixEyePointCameraCoordFixUp(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  //rotate around y-axis
  Eigen::Matrix<T, 3, 1> upVector { 0.0, 1.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, upVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, upVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, upVector, in_HorizontalAngle);

  //rotate around x-axis
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.xVector, in_VerticalAngle);

  //rotate around z-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.internalData.zVector, -in_RoundAngle);

  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixEyePointGlobalCoord(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  //rotate around y-axis
  Eigen::Matrix<T, 3, 1> yVector { 0.0, 1.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, yVector, in_HorizontalAngle);

  //rotate around x-axis
  Eigen::Matrix<T, 3, 1> xVector { 1.0, 0.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, xVector, in_VerticalAngle);

  //rotate around z-axis
  Eigen::Matrix<T, 3, 1> zVector { 0.0, 0.0, 1.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, zVector, -in_RoundAngle);

  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixLookAtCameraCoord(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  const Eigen::Matrix<T, 3, 1> fixed_look_at { -io_Camera.internalData.focusLength * io_Camera.internalData.zVector + io_Camera.eyePoint };

  Eigen::Matrix<T, 3, 1> arm { io_Camera.internalData.focusLength * io_Camera.internalData.zVector };

  //rotate around y-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.yVector, in_HorizontalAngle);
  rotateVector<T>(&arm, arm, io_Camera.yVector, in_HorizontalAngle);

  //rotate around x-axis
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&arm, arm, io_Camera.xVector, in_VerticalAngle);

  //rotate around z-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&arm, arm, io_Camera.internalData.zVector, -in_RoundAngle);

  io_Camera.eyePoint = fixed_look_at + arm;
  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixLookAtCameraCoordFixUp(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  const Eigen::Matrix<T, 3, 1> fixed_look_at { -io_Camera.internalData.focusLength * io_Camera.internalData.zVector + io_Camera.eyePoint };

  Eigen::Matrix<T, 3, 1> arm { io_Camera.internalData.focusLength * io_Camera.internalData.zVector };

  //rotate around y-axis
  Eigen::Matrix<T, 3, 1> upVector { 0.0, 1.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, upVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, upVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, upVector, in_HorizontalAngle);
  rotateVector<T>(&arm, arm, upVector, in_HorizontalAngle);

  //rotate around x-axis
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&arm, arm, io_Camera.xVector, in_VerticalAngle);

  //rotate around z-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&arm, arm, io_Camera.internalData.zVector, -in_RoundAngle);

  io_Camera.eyePoint = fixed_look_at + arm;
  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixLookAtCameraCoordFixUp(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera, const Eigen::Matrix<T, 3, 1>& in_Up)
{
  const Eigen::Matrix<T, 3, 1> fixed_look_at { -io_Camera.internalData.focusLength * io_Camera.internalData.zVector + io_Camera.eyePoint };

  Eigen::Matrix<T, 3, 1> arm { io_Camera.internalData.focusLength * io_Camera.internalData.zVector };

  //rotate around y-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, in_Up, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, in_Up, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, in_Up, in_HorizontalAngle);
  rotateVector<T>(&arm, arm, in_Up, in_HorizontalAngle);

  //rotate around x-axis
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, io_Camera.xVector, in_VerticalAngle);
  rotateVector<T>(&arm, arm, io_Camera.xVector, in_VerticalAngle);

  //rotate around z-axis
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, io_Camera.internalData.zVector, -in_RoundAngle);
  rotateVector<T>(&arm, arm, io_Camera.internalData.zVector, -in_RoundAngle);

  io_Camera.eyePoint = fixed_look_at + arm;
  setPlane<T>(io_Camera);
}

template<typename T>
void rotateFixLookAtGlobalCoord(T in_HorizontalAngle, T in_VerticalAngle, T in_RoundAngle, Camera<T>& io_Camera)
{
  const Eigen::Matrix<T, 3, 1> fixed_look_at { -io_Camera.internalData.focusLength * io_Camera.internalData.zVector + io_Camera.eyePoint };

  Eigen::Matrix<T, 3, 1> arm { io_Camera.internalData.focusLength * io_Camera.internalData.zVector };

  //rotate around y-axis
  Eigen::Matrix<T, 3, 1> yVector { 0.0, 1.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, yVector, in_HorizontalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector,yVector, in_HorizontalAngle);
  rotateVector<T>(&arm, arm, yVector, in_HorizontalAngle);

  //rotate around x-axis
  Eigen::Matrix<T, 3, 1> xVector { 1.0, 0.0, 0.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, xVector, in_VerticalAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, xVector, in_VerticalAngle);
  rotateVector<T>(&arm, arm, xVector, in_VerticalAngle);

  //rotate around z-axis
  Eigen::Matrix<T, 3, 1> zVector { 0.0, 0.0, 1.0 };
  rotateVector<T>(&io_Camera.xVector, io_Camera.xVector, zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.yVector, io_Camera.yVector, zVector, -in_RoundAngle);
  rotateVector<T>(&io_Camera.internalData.zVector, io_Camera.internalData.zVector, zVector, -in_RoundAngle);
  rotateVector<T>(&arm, arm, zVector, -in_RoundAngle);

  io_Camera.eyePoint = fixed_look_at + arm;
  setPlane<T>(io_Camera);
}

#endif

