add_library(aksharanet_detection STATIC detector.cpp)
target_include_directories(aksharanet_detection PUBLIC include)
target_link_libraries(aksharanet_detection
    PUBLIC  ${OpenCV_LIBS}
    PRIVATE onnxruntime::onnxruntime aksharanet_warnings
)
