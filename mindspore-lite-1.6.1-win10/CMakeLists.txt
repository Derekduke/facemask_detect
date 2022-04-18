cmake_minimum_required(VERSION 3.12)  # cmake版本要求最低3.12
project(Demo)  # 工程名为Demo

# gcc 版本要求大于等于7.3.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/)  # 添加头文件寻找路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/data/)  # 添加头文件寻找路径

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/lib)  # 添加库文件寻找路径

add_executable(demo main.cc)  # 编译生成demo执行程序

# 声明demo执行程序需要链接的库，mindspore-lite为MindSpore Lite推理框架的动态库
target_link_libraries(
        demo
        mindspore-lite
        pthread
        ssp
)