set(ZNZ_SRC znzlib.h znzlib.c)

find_package(zlib)
if (ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
else()
  include_directories(../zlib/include)
endif()

add_library(znzlib ${ZNZ_SRC})
target_link_libraries(znzlib zlib)

