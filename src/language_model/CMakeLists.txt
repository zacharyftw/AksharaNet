add_library(aksharanet_language_model STATIC language_model.cpp)
target_include_directories(aksharanet_language_model PUBLIC include)
target_link_libraries(aksharanet_language_model
    PRIVATE ICU::uc ICU::i18n kenlm aksharanet_warnings
)
