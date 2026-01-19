;==================================================================
;                     USER INTERFACE 
;                   MAIN DRIVER SCRIPT
;==================================================================

;===================================================================  
; Calculate potential temperature
;===================================================================  
  input_directory = "/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid/"
  input_filename_t  = "t.nc"
  input_dataset_t  = addfile(input_directory + input_filename_t, "r")
  input_filename_q  = "q.nc"
  input_dataset_q  = addfile(input_directory + input_filename_q, "r")
  output_directory = "/work/b08209033/DATA/tmp/"
  output_filename  = "equivalent_potential_temperature.nc"
  system("/bin/rm -f " + output_directory + output_filename)
  output_dataset  = addfile(output_directory + output_filename, "c")
	
  t4   = input_dataset_t->t(:,:,:,:)
  q4   = input_dataset_q->q(:,:,:,:)
  p1   = input_dataset_t->plev(:) ; assume metadata is identical for all input
  ept4 = pot_temp_equiv(p1, t4, q4, 1, "q")
;===================================================================  
; Write to netCDF
;===================================================================
  setfileoption(output_dataset,"DefineMode",True)

  time  = input_dataset_t->time(:)
  plev  = input_dataset_t->plev(:)
  lat   = input_dataset_t->lat(:)
  lon   = input_dataset_t->lon(:)
  ntim  = dimsizes(time)
	nlev  = dimsizes(plev)
	nlat  = dimsizes(lat)
	nlon  = dimsizes(lon)

  dimNames = (/"time", "lat", "lon", "plev"/)  
	dimSizes = (/ -1   ,  nlat,  nlon,  nlev /) 
	dimUnlim = (/ True , False, False, False/)   
	filedimdef(output_dataset, dimNames, dimSizes, dimUnlim)

  filevardef(output_dataset, "time" ,typeof(time) ,getvardims(time)) 
  filevardef(output_dataset, "plev" ,typeof(plev) ,getvardims(plev))                           
  filevardef(output_dataset, "lat"  ,typeof(lat)  ,getvardims(lat))                          
  filevardef(output_dataset, "lon"  ,typeof(lon)  ,getvardims(lon))                          
  filevardef(output_dataset, "ept"   ,typeof(ept4)  ,getvardims(ept4))

  filevarattdef(output_dataset, "ept"  ,ept4)
  filevarattdef(output_dataset, "time" ,time)
  filevarattdef(output_dataset, "plev" ,plev)
  filevarattdef(output_dataset, "lat"  ,lat)
  filevarattdef(output_dataset, "lon"  ,lon)

  output_dataset->time  = (/time/)     
  output_dataset->plev  = (/plev/)
  output_dataset->lat   = (/lat/)
  output_dataset->lon   = (/lon/) 
  output_dataset->ept   = (/ept4/)

  setfileoption(output_dataset,"DefineMode",False)