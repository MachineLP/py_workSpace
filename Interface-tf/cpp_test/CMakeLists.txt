cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(example  CXX)
SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")
find_package(OpenCV)
include_directories($(OpenCV_INCLUDE_DIRS))

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)


# protobuf and grpc
# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# find_package(Protobuf REQUIRED)

set(PROTOS  ${CMAKE_CURRENT_SOURCE_DIR}/protos/image_analyzing.proto)

set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-src)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})
include_directories(${PROTO_SRC_DIR})

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR} ${PROTOS})
# grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_SRC_DIR} ${PROTOS})


find_package(CUDA)
message(${CUDA_LIBRARIES})

# 服务端
set(app_s_SRCS
./src/tf_pb.cpp
${PROTO_SRCS}
${GRPC_SRCS}
)

add_executable(app_s ${app_s_SRCS})



target_link_libraries(app_s
tensorflow_cc
tensorflow_framework
opencv_core
opencv_imgproc
opencv_imgcodecs
curlpp
curl
log4cplus
pthread
${CUDA_LIBRARIES}
${OpenCV_LIBS}
)

