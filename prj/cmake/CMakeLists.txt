cmake_minimum_required(VERSION 2.8)
project(TestDnn)

set(ROOT_DIR ${CMAKE_SOURCE_DIR}/../..)
set(3RD_DIR ${ROOT_DIR}/3rd)

include_directories(${3RD_DIR}/Cpl/src)

file(GLOB_RECURSE TEST_DNN_SRC ${ROOT_DIR}/src/TestDnn/*.cpp)
add_executable(TestDnn ${TEST_DNN_SRC})
target_link_libraries(TestDnn -lpthread)