add_library(aksharanet_recognition STATIC recognizer.cpp)
target_include_directories(aksharanet_recognition PUBLIC include)
target_link_libraries(aksharanet_recognition
    PUBLIC  ${OpenCV_LIBS}
    PRIVATE onnxruntime::onnxruntime aksharanet_warnings
)
