set(ZNZ_SRC znzlib.h znzlib.c)

include_directories(../zlib/include)

add_library(znzlib ${ZNZ_SRC})
target_link_libraries(znzlib zlib)

