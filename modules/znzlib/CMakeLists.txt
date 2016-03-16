set(ZNZ_SRC znzlib.h znzlib.c)

find_package(zlib)
include_directories(${ZLIB_INCLUDE_DIRS})

add_library(znzlib ${ZNZ_SRC})
target_link_libraries(znzlib zlib)

