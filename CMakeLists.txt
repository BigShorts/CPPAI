cmake_minimum_required(VERSION 3.5)
project(CPPAI)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")
add_executable(CPPAI ${SRC_FILES})

execute_process(COMMAND ./CompileLLama.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(CPPAI PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libcommon.a
    ${CMAKE_CURRENT_SOURCE_DIR}/build/libllama.so
    sqlite3
)

file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/models" "${CMAKE_CURRENT_SOURCE_DIR}/build/models" SYMBOLIC)
file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/articles" "${CMAKE_CURRENT_SOURCE_DIR}/build/articles" SYMBOLIC)
file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/summaries" "${CMAKE_CURRENT_SOURCE_DIR}/build/summaries" SYMBOLIC)

target_include_directories(CPPAI PUBLIC "lib/llama.cpp")
