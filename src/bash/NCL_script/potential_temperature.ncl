;==================================================================
;                     USER INTERFACE 
;                   MAIN DRIVER SCRIPT
;==================================================================

;===================================================================  
; Calculate potential temperature
;===================================================================  
  input_directory = "/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid/"
  input_filename  = "t.nc"
  input_dataset  = addfile(input_directory + input_filename, "r")
  output_directory = "/work/b08209033/DATA/tmp/"
  output_filename  = "potential_temperature.nc"
  system("/bin/rm -f " + output_directory + output_filename)
  output_dataset  = addfile(output_directory + output_filename, "c")
	
  t4   = input_dataset->t(:,:,:,:)
  p1   = input_dataset->plev(:)
  pt4  = pot_temp(p1 ,t4 , 1, False)
;===================================================================  
; Write to netCDF
;===================================================================
  setfileoption(output_dataset,"DefineMode",True)

  time  = input_dataset->time(:)
  plev  = input_dataset->plev(:)
  lat   = input_dataset->lat(:)
  lon   = input_dataset->lon(:)
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
  filevardef(output_dataset, "pt"   ,typeof(pt4)  ,getvardims(pt4))

  filevarattdef(output_dataset, "pt"   ,pt4)
  filevarattdef(output_dataset, "time" ,time)
  filevarattdef(output_dataset, "plev" ,plev)
  filevarattdef(output_dataset, "lat"  ,lat)
  filevarattdef(output_dataset, "lon"  ,lon)

  output_dataset->time  = (/time/)     
  output_dataset->plev  = (/plev/)
  output_dataset->lat   = (/lat/)
  output_dataset->lon   = (/lon/) 
  output_dataset->pt    = (/pt4/)

  setfileoption(output_dataset,"DefineMode",False)