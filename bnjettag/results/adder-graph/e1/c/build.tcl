open_project -reset prj_c
set_top myproject
add_files myproject.cpp
add_files -tb tb.cpp
open_solution -reset sol1
set_part xcvu13p-flga2577-2-e
create_clock -period 2.5
csim_design
csynth_design
config_export -format ip_catalog -rtl verilog
export_design -flow syn
exit
