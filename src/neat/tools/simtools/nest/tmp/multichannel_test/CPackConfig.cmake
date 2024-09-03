# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test;/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/Users/wybo/miniconda3/envs/nest-multiscale/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "multichannel_test_module built using CMake")
set(CPACK_GENERATOR "TGZ")
set(CPACK_INSTALL_CMAKE_PROJECTS "/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test;multichannel_test_module;ALL;/")
set(CPACK_INSTALL_PREFIX "/Users/wybo/miniconda3/envs/nest-multiscale")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "multichannel_test_module 1.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "multichannel_test_module 1.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJDUMP_EXECUTABLE "/Library/Developer/CommandLineTools/usr/bin/objdump")
set(CPACK_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.4.sdk")
set(CPACK_OUTPUT_CONFIG_FILE "/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/Users/wybo/miniconda3/envs/nest-multiscale/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NEST Module multichannel_test_module")
set(CPACK_PACKAGE_FILE_NAME "multichannel_test_module-1.0-Darwin")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "multichannel_test_module 1.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "multichannel_test_module 1.0")
set(CPACK_PACKAGE_NAME "multichannel_test_module")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "NEST Initiative (http://www.nest-initiative.org/)")
set(CPACK_PACKAGE_VERSION "1.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_RESOURCE_FILE_LICENSE "/Users/wybo/miniconda3/envs/nest-multiscale/share/cmake-3.26/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/Users/wybo/miniconda3/envs/nest-multiscale/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/Users/wybo/miniconda3/envs/nest-multiscale/share/cmake-3.26/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES "\\.gitignore;\\.git/;\\.travis\\.yml;/build/;/_CPack_Packages/;CMakeFiles/;cmake_install\\.cmake;Makefile.*;CMakeCache\\.txt;CPackConfig\\.cmake;CPackSourceConfig\\.cmake")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test/CPackSourceConfig.cmake")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "multichannel_test_module")
set(CPACK_SYSTEM_NAME "Darwin")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Darwin")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/Users/wybo/Code/NEAT_public/neat/tools/simtools/nest/tmp/multichannel_test/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
