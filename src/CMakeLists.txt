set(GPUE_SUBDIRS    vortex;
                    io;
                    simulator;
                    utils;
                    tests;
                    bindings;
)
foreach(MOD ${GPUE_SUBDIRS})
    add_subdirectory(${MOD})
endforeach()
