#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import xarray as xr
import yaml


#  # set up folder structure




# change yaml location here
with open(r"/home/mwegmann/g2s/notebooks/folder_gap_filling.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)





# defining folders
input_folder=directories["g2s_input_folder"]

output_folder=directories["g2s_output_folder"]


#  # load cosmo files




cosmo=xr.open_mfdataset(input_folder+"cosmo2_epfl_lakes_*.nc")


# # define variables of interest




variables_of_interest=["T_2M","RELHUM_2M","TOT_PREC","PS","GLOB","CLCT","U","V"]





# lat lon location of the platform
# 46.50027102909318, 6.660955285962016


# # extract relevant data space




# x_1 -2.29 is the respective gridbox for lon_platform
# y_1 -0.45 is the respective gridbox for lat_platform
# z_3 is 10 meters for the wind 
cosmo_lexplore_gapfilling=cosmo[variables_of_interest].sel(x_1=-2.29,y_1=-0.45,z_3=10,time=slice("2020-06-01","2023-05-31"))


# # write data




cosmo_lexplore_gapfilling.to_netcdf(output_folder+"cosmo_lexplore_gapfilling.nc")







