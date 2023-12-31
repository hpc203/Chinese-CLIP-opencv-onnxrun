project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/usr/local/include/opencv4"
        PUBLIC "/opt/onnxruntime-linux-x64-1.11.1/include"
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        "/usr/local/lib/libopencv_imgcodecs.so.4.8.0"
        "/usr/local/lib/libopencv_highgui.so.4.8.0"
        "/usr/local/lib/libopencv_imgproc.so.4.8.0"
        "/usr/local/lib/libopencv_core.so.4.8.0"
        "/usr/local/lib/libopencv_dnn.so.4.8.0"
        "/opt/onnxruntime-linux-x64-1.11.1/lib/libonnxruntime.so.1.11.1"
        )