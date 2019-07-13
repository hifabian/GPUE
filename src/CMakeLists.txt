set(GPUE_SUBDIRS    vortex_lib;
                    io;
)
foreach(MOD ${GPUE_SUBDIRS})
    add_subdirectory(${MOD})
endforeach()
