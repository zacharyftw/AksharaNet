add_library(aksharanet_decoding STATIC decoder.cpp)
target_include_directories(aksharanet_decoding PUBLIC include)
target_link_libraries(aksharanet_decoding
    PUBLIC  aksharanet_recognition
    PRIVATE ICU::uc ICU::i18n kenlm aksharanet_warnings
)
