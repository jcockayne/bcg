cmake_minimum_required (VERSION 2.6)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project (bcg C CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package( PkgConfig )

pkg_check_modules(EIGEN3 REQUIRED eigen3)

include_directories(include ${EIGEN3_INCLUDE_DIRS})
file(GLOB SOURCES "src/*.cpp")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DENABLE_LOG_ERROR -DENABLE_LOG_WARN -DENABLE_LOG_INFO")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DENABLE_LOG_ERROR -DENABLE_LOG_WARN -DENABLE_LOG_INFO")

add_library(bcg STATIC ${SOURCES})
target_compile_features(bcg PRIVATE cxx_range_for)
target_link_libraries(bcg)