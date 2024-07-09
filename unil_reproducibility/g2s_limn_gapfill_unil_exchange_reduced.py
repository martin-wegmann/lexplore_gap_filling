#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
import os
import math
import metpy
import datetime
import matplotlib.pyplot as plt 
import scipy.stats as sstats
from scipy.stats.sampling import DiscreteAliasUrn
from matplotlib import cm
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import IPython.display
import json
import sys
import yaml
from random import randrange
import metpy.calc as mpcalc
from metpy.units import units
from functions_gapfill import *
import gstools as gs
import statsmodels.api as sm


# # Gapfilling the limnology data of the LéXPLORE platform

# In this notebook we use the G2S server with Direct Sampling approach (https://gaia-unil.github.io/G2S/briefOverview.html) to fill the data gaps of the post-processed selection of limnology data of the LéXPLORE platform (https://gitlab.renkulab.io/lexplore).
# 
# We use already 3 hourly aggregated values of three platforms:
# 
# * The Thermister Chain (TChain from here onwards)
# 
# * The Thetis Multispectral thingy (Thetis from here onwards)
# 
# * The lake profiler Idronaut (Idronaut from here onwards)
# 
# and we fill gaps for the following variables on 3 hour resolution:
# 
# * Water temperature (Tchain, Thetis, Idronaut)
# 
# * Chlorophyll A (Thetis, Idronaut)
# 
# * Dissolved Oxygen (Thetis, Idronaut)
# 
# * Oxygen Saturation (Thetis, Idronaut)
# 
# 
# on 38 levels for each data set. 
# 
# To do so, we use independent data as co-variates, namely the other two sensor platforms as well as simulated water temperature from https://www.alplakes.eawag.ch/.

# ## Activate G2S server

# In[2]:


#!pip install G2S libtiff --quiet
from g2s import g2s
g2s('--version')


# In[4]:


get_ipython().system('g2s server -d')


# ## folder setup

# In[5]:


# change yaml location here
with open(r"/home/martinw/gapfill/notebooks/folder_gap_filling_giub.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)


# In[6]:


# defining folders
input_folder=directories["g2s_input_folder"]

output_folder=directories["g2s_output_folder"]

plots_folder=directories["g2s_plot_folder"]

recs_folder=directories["g2s_reconstructions_folder"]

scripts_folder=directories["scripts_folder"]


# ## read in postprocessed input data

# check notebooks X and Y to see how this data was created.

# ### TChain

# In[7]:


tchain=xr.open_dataset(input_folder+"tchain_3hr_g2s.nc")



# ## define L3 boundaries

# In[16]:


time_resolution_hr=3


# In[17]:


timestepsinday=int(24/time_resolution_hr)


# In[18]:


max_day_gap=5


# In[19]:


max_timesteps_tofill=timestepsinday*max_day_gap


# ## Fill Tchain Data
## you can make the data set smaller here if you want

varname="temp"

temp_tchain_data=tchain[varname].copy()
data_original=temp_tchain_data.copy()

### adding some gaps, for the sake of it we just introduce 5 % new gaps and each gap is 8 timesteps long (8 timesteps = 24 hours)
### in reality we do that several times over different gap sizes and call it "test_runs" in order to put the gaps somewhere else every time

#percent_list=[5]
gap_amount_list=[8]
selector_list=[1]
#N = 25
#test_runs=10

percent=5
obs_in_day=timestepsinday


gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day,gap_amount=gap_amount_list)

gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list,selector=selector_list)

# #### reconstruction phase

#### This is how you create the reconstruction
N=10 # you can play around with that value

timeofday = data_original.time.dt.hour.values

depth_dim, time_dim = data_original.shape

depth_linear = np.transpose(np.tile(data_original.depth.data,(time_dim,1)))
    
depth_inverse = 1/depth_linear

## creating the depth penalization based on layer variance in time
    # where do we have more than 50% nans
mask_var=data_original.isnull().sum(dim="time")>(data_original.data.shape[1]/2) # this needs to be changed for IDRONAUT AND THETIS
    # create the variance over time
da_var_depth=data_original.var(dim="time")
    # where we have more than 50% nans, we dont trust the variance and put missing values
da_var_depth[mask_var]=np.nan
    #we fill these missing values with linear interpolated values
da_var_depth["depth"]=da_var_depth["depth"]*-1
da_var_depth_int=da_var_depth.interpolate_na(dim="depth", method="linear")
da_var_depth_int["depth"]=data_original["depth"]

depth_variance=np.transpose(np.tile(da_var_depth_int.data,(time_dim,1)))


sin_calendar = sin_costfunction(time_dim ,daily_timesteps = timestepsinday)
cos_calendar = cos_costfunction(time_dim ,daily_timesteps = timestepsinday)

sin_2D = np.tile(sin_calendar, (depth_dim,1))
cos_2D = np.tile(cos_calendar, (depth_dim,1))

ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
dt = [0,0,0,0]


################ maybe you need to change the output here to the new ensemble_QS syntax


stacked = ensemble_QS(N = N,ti=ti, di=di,dt=dt,k=1.2, n=50,j=0.5,ki=None)


simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time}) 

################         

### lets use another interpolation method as comparison
simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")

### check variogram output between the two methods
### now this is obviously for 5% missing values and gap length 8

depth_array=[-10] # you can play around with that value or add more values

for single_depth in depth_array:
	print("Looking at depth: "+str(single_depth ))
	qs_mean=simulations.sel(depth=single_depth,method="nearest").mean(dim="realizations").squeeze()
	qs_median=simulations.sel(depth=single_depth,method="nearest").median(dim="realizations").squeeze()
	print("load the mean and median data")
	qs_mean=qs_mean.load()
	qs_median=qs_median.load()

	sims = np.array([qs_mean,qs_median,simulations_lin.sel(depth=single_depth,method="nearest").squeeze()])
	bin_corrector=24/obs_in_day
	print(sims.shape)
	
	print("now computing variogram")
	print(datetime.datetime.now())
	bin_centers,gamma_obs, gamma_sim_list = compare_variograms_nothreads(data_original,
	  sims, 
	   gap_indices = None,
	  bin_number = int(96/int(bin_corrector)))
	
	rmse_var_list=[]
	for sim in range(sims.shape[0]):
		rmse = np.round(np.sqrt(np.nanmean((gamma_obs- gamma_sim_list[sim])**2)),4)
		rmse_var_list.append(rmse)
	member_mean=rmse_var_list[0]
	member_median=rmse_var_list[1]
	lin_mean=rmse_var_list[2]

print(member_mean)
print(member_median)
print(lin_mean)

