execute_process(COMMAND "/home/cc/ee106b/sp25/class/ee106b-aan/ee106b_workspace/106b-projects/proj1_pkg/build/baxter_pykdl/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/cc/ee106b/sp25/class/ee106b-aan/ee106b_workspace/106b-projects/proj1_pkg/build/baxter_pykdl/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
