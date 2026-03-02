include(FetchContent)

if(AKSHARANET_SYSTEM_KENLM)
    find_package(kenlm REQUIRED)
else()
    FetchContent_Declare(
        kenlm
        GIT_REPOSITORY https://github.com/kpu/kenlm.git
        GIT_TAG        master
    )
    set(COMPILE_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kenlm)
endif()
