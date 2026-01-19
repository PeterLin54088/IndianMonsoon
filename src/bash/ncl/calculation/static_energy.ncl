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
  input_filename_z  = "z.nc"
  input_dataset_z  = addfile(input_directory + input_filename_z, "r")
  output_directory = "/work/b08209033/DATA/tmp/"
  output_filename  = "static_energy.nc"
  system("/bin/rm -f " + output_directory + output_filename)
  output_dataset  = addfile(output_directory + output_filename, "c")
	
  t   = input_dataset_t->t(:,:,:,:)
  q   = input_dataset_q->q(:,:,:,:)
  z   = input_dataset_z->z(:,:,:,:)
  z  := z/9.78
  SE  = static_energy_atm(t, q, z, (/1,0 /))
  MSE = SE[0]
  DSE = SE[1]
  LSE = SE[2]
  delete(SE)
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
  filevardef(output_dataset, "dse"   ,typeof(DSE)  ,getvardims(DSE))
  filevardef(output_dataset, "lse"   ,typeof(LSE)  ,getvardims(LSE))
  filevardef(output_dataset, "mse"   ,typeof(MSE)  ,getvardims(MSE))

  filevarattdef(output_dataset, "dse"  ,DSE)
  filevarattdef(output_dataset, "lse"  ,LSE)
  filevarattdef(output_dataset, "mse"  ,MSE)
  filevarattdef(output_dataset, "time" ,time)
  filevarattdef(output_dataset, "plev" ,plev)
  filevarattdef(output_dataset, "lat"  ,lat)
  filevarattdef(output_dataset, "lon"  ,lon)

  output_dataset->time  = (/time/)     
  output_dataset->plev  = (/plev/)
  output_dataset->lat   = (/lat/)
  output_dataset->lon   = (/lon/) 
  output_dataset->dse    = (/DSE/)
  output_dataset->lse    = (/LSE/)
  output_dataset->mse    = (/MSE/)

  setfileoption(output_dataset,"DefineMode",False)