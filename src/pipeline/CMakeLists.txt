add_library(aksharanet_pipeline STATIC pipeline.cpp)
target_include_directories(aksharanet_pipeline PUBLIC include)
target_link_libraries(aksharanet_pipeline
    PUBLIC
        aksharanet_preprocessing
        aksharanet_detection
        aksharanet_recognition
        aksharanet_decoding
        aksharanet_language_model
    PRIVATE aksharanet_warnings
)
