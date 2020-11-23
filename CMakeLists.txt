cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/include/numpy/ )
set(PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libpython3.6m.so)
find_package(PythonLibs REQUIRED)

if (NOT PythonLibs_FOUND)
	message(FATAL_ERROR "PYTHON LIBS not found.")
else()
	message("PYTHON LIBS were found!")
	message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
endif()

#set(OpenCV_DIR /usr/local/opencv2.4.13/share/OpenCV)
message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV  QUIET)
if(NOT OpenCV_FOUND)
	message("OpenCV >3.0 not found.")
endif()

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)



# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
		/usr/include/python3.6/
		/usr/include/python3.6m/
		/usr/include/numpy/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(camera_model SHARED
    src/camera_models/chessboard/Chessboard.cc
    src/camera_models/calib/CameraCalibration.cc
    src/camera_models/camera_models/Camera.cc
    src/camera_models/camera_models/CameraFactory.cc
    src/camera_models/camera_models/CostFunctionFactory.cc
    src/camera_models/camera_models/PinholeCamera.cc
    src/camera_models/camera_models/CataCamera.cc
    src/camera_models/camera_models/EquidistantCamera.cc
    src/camera_models/camera_models/ScaramuzzaCamera.cc
    src/camera_models/sparse_graph/Transform.cc
    src/camera_models/gpl/gpl.cc
    src/camera_models/gpl/EigenQuaternionParameterization.cc src/toolsForModel.cpp include/toolsForModel.h)

target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})


ADD_LIBRARY(MyVio SHARED
	src/System.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
	src/feature_tracker.cpp

    src/utility/utility.cpp
	src/utility/Random.cpp
	src/utility/Timestamp.cpp

    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp

    src/backend/vertex.cc
    src/backend/edge.cc
    src/backend/problem.cc
    src/backend/vertex_pose.cc
    src/backend/edge_reprojection.cc
    src/backend/edge_imu.cc
    src/backend/edge_prior.cpp
    src/backend/loss_function.cc
    src/backend/imu_integration.cc
	src/Conversion.cc

	src/InstanceSeg.cpp
	src/Instance_Object.cpp
	src/MSG.cpp
    src/toolsForModel.cpp
	include/Pose_for_Objects.h src/Pose_for_Objects.cpp)

target_link_libraries(MyVio  
  ${OpenCV_LIBS}  
  ${CERES_LIBRARIES}
	${Pangolin_LIBRARIES}
		/usr/lib/x86_64-linux-gnu/libpython3.6m.so
		/usr/include/numpy/
  camera_model)

add_executable(run_KITTI test/run_KITTI.cpp)
target_link_libraries(run_KITTI
  MyVio
 -lpthread)

#add_executable(testCurveFitting test/CurveFitting.cpp)
#target_link_libraries(testCurveFitting MyVio)

## 测试C与python之间通信；
#add_executable(CInsSegTest test/CpInsSegTest.cpp)
#target_link_libraries(CInsSegTest MyVio)

##测试实例分割网络整体；
#add_executable(InstanceTest test/InTest.cpp)
#target_link_libraries(InstanceTest MyVio)

#add_executable(ImgTest test/ImgtypeTest.cpp)
#target_link_libraries(ImgTest MyVio)

## 测试系统读取参数等等；
add_executable(SystemTest test/TestSystemOriginal.cpp)
target_link_libraries(SystemTest MyVio)








