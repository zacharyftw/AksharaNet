add_library(aksharanet_preprocessing STATIC preprocessor.cpp)
target_include_directories(aksharanet_preprocessing PUBLIC include)
target_link_libraries(aksharanet_preprocessing
    PUBLIC  ${OpenCV_LIBS}
    PRIVATE aksharanet_warnings
)
