;---------------------------------------------
; wkSpaceTime.ncl
;---------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;
; This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"

;==================================================================
;                     USER INTERFACE 
;                   MAIN DRIVER SCRIPT
;==================================================================
  file1      = addfile ("/work/b08209033/DATA/IndianMonsoon/ERA5/sparse_grid/sparse_v.nc", "r")
  file2      = addfile ("/work/b08209033/DATA/IndianMonsoon/ERA5/sparse_grid/sparse_mse.nc", "r")  
  caseName   = "ERA5"
  diro       = "/home/b08209033/IndianMonsoon/img/NCL_wkSpaceTime/v*h/"
  
  latBound   = 15
  spd        = 1
  nDayWin    = 96
  nDaySkip   = -65
  
  plevel = 100000
  varName  = "meridional_mse_flux"
  varName := varName+" (at "+plevel/100+" hPa)"
  v  = file1->v(:,{plevel},{latBound:-latBound},:)
  mse  = file2->mse(:,{plevel},{latBound:-latBound},:)
  vh = v*mse

  vh!0 = "time"
  vh!1 = "lat"
  vh!2 = "lon"
  vh&time = file1->time
  vh&lat  = file1->lat({latBound:-latBound})
  vh&lon  = file1->lon
  vh@units = "m**2 s**-1"
  vh@long_name = "multiplication of moist static energy and meridional wind"
  vh@standard_name = varName
  vh@plev = plevel

  opt      = True
  opt@debug       = True   
  opt@pltType     = "png"        ; send graphics to PNG file
  opt@pltTitle    = caseName+" "+varName+" LOG[Power: "+latBound+"S-"+latBound+"N]"
  opt@Fig_3a      = fspan(0.6, 2.0, 15)
  opt@Fig_3b      = fspan(0.6, 2.0, 15)
  

  wkSpaceTime (vh,  diro, caseName, varName             \
              ,latBound, spd , nDayWin, nDaySkip, opt  )