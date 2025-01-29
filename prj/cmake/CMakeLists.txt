cmake_minimum_required(VERSION 3.10)
project(TestDnn)

set(ROOT_DIR ${CMAKE_SOURCE_DIR}/../..)
set(3RD_DIR ${ROOT_DIR}/3rd)

include_directories(${3RD_DIR}/Cpl/src)

set(SIMD_AVX512 ON CACHE BOOL "" FORCE)
set(SIMD_AVX512VNNI ON CACHE BOOL "" FORCE)
set(SIMD_AMXBF16 ON CACHE BOOL "" FORCE)
set(SIMD_TEST OFF CACHE BOOL "" FORCE)
set(SIMD_SHARED ON CACHE BOOL "" FORCE)
set(SIMD_GET_VERSION OFF CACHE BOOL "" FORCE)
set(SIMD_TOOLCHAIN ${CMAKE_CXX_COMPILER})
set(SIMD_INFO OFF CACHE BOOL "" FORCE)
set(SIMD_PERF OFF CACHE BOOL "" FORCE)
set(SIMD_SYNET ON "" FORCE)
set(SIMD_PYTHON OFF "" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include_directories(${3RD_DIR}/Simd/src)
add_subdirectory(${3RD_DIR}/Simd/prj/cmake Simd)

set(DNNL_ROOT_DIR ${3RD_DIR}/oneDNN)
set(DNNL_BUILD_DIR ${CMAKE_BINARY_DIR}/oneDNN)
set(DNNL_LIBS ${DNNL_BUILD_DIR}/libdnnl.so)

set(DNNL_BUILD_OPTIONS 
	-DCMAKE_BUILD_TYPE="Release"
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${DNNL_BUILD_DIR}"
	)

file(MAKE_DIRECTORY ${DNNL_BUILD_DIR})
add_custom_command(
	OUTPUT ${DNNL_LIBS}
	COMMAND cmake ${DNNL_ROOT_DIR} ${DNNL_BUILD_OPTIONS} && make -j16
	POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${DNNL_LIBS} ${CMAKE_BINARY_DIR} 
	WORKING_DIRECTORY ${DNNL_BUILD_DIR})
	

add_custom_target(make_dnnl DEPENDS ${DNNL_LIBS})
include_directories(${DNNL_ROOT_DIR}/include ${DNNL_BUILD_DIR}/include)

file(GLOB_RECURSE TEST_DNN_SRC ${ROOT_DIR}/src/TestDnn/*.cpp)
add_executable(TestDnn ${TEST_DNN_SRC})
add_dependencies(TestDnn make_dnnl)
target_link_libraries(TestDnn Simd ${DNNL_LIBS} -lpthread)