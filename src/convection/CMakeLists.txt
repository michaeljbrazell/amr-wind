target_sources(${amr_wind_exe_name}
   PRIVATE
      #C++
      incflo_MAC_bcs.cpp
      incflo_godunov_ppm.cpp
      incflo_mol_fluxes_eb.cpp
      incflo_MAC_projection.cpp	
      incflo_correct_small_cells.cpp
      incflo_godunov_plm.cpp	
      incflo_godunov_predict.cpp
      incflo_mol_predict.cpp
      incflo_compute_advection_term.cpp
      incflo_godunov_advection.cpp
      incflo_mol_fluxes.cpp
      incflo_mol_predict_eb.cpp
   )

target_include_directories(${amr_wind_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
