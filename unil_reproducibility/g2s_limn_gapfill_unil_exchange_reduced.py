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


gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,
			gap_percent=percent,gap_length=obs_in_day,gap_amount=gap_amount_list)

gapped_data=create_gapped_ts_2D(da=data_original,
gap_locations=gap_locations,depth_level_index=depth_level_indices,
gap_length=gap_amount_list[0],selector=selector_list[0])

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

# ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
# di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)


ti = np.stack([gapped_data.data, sin_2D, cos_2D],axis = 2)
di = np.stack([gapped_data.data,sin_2D, cos_2D],axis = 2)
dt = [0,0,0]


################ maybe you need to change the output here to the new ensemble_QS syntax
kernel_thicknesses = [1,2,3,5,7,10]
sim_mean_rmses = {}
sim_median_rmses = {}
simulations_dic = {}
indices_dic = {}

for kernel_thickness in kernel_thicknesses:

    # simulations_dic[kernel_thickness] = xr.open_dataarray(join(output_folder,f"tchain_{varname}_gapfilled_{kernel_thickness}.nc"))

    # kzeros = np.zeros((1,time_dim,1))    
    ki = np.ones((kernel_thickness,time_dim))
    # ki = np.dstack([kzeros,kones,kzeros])

    stacked,index = ensemble_QS(sa = 'tesla-k20c.gaia.unil.ch',
                    N = N,ti=ti, di=di,dt=dt,k=1.2, n=50,j=0.5,ki=ki)


    simulations = xr.DataArray(data =stacked[:,:,:,0],
                    coords = {'realizations':np.arange(1,stacked.shape[0]+1),
                    'depth':data_original.depth.data,
                    'time':gapped_data.time}) 

    index = xr.DataArray(data =index[:,:,:],
                    coords = {'realizations':np.arange(1,index.shape[0]+1),
                    'depth':data_original.depth.data,
                    'time':gapped_data.time}) 
    index = xr.where(index == 0, np.nan, index)
    # simulations.to_netcdf(join(output_folder,f"tchain_{varname}_gapfilled_{kernel_thickness}.nc"))


    simulations_dic[kernel_thickness] = simulations
    indices_dic[kernel_thickness] = index
#%%
for kernel_thickness in kernel_thicknesses:
    simulations = simulations_dic[kernel_thickness]    

    # index = indices_dic[kernel_thickness]

    year =   2021
    start_month = 1
    end_month = 12

    # plot_MPS_ensembles_2D(original = tchain.temp,
    #                 simulation= simulations,
    #                 year = year,
    #                 start_month = start_month,
    #                 end_month = end_month,
    #                 suptitle = "Tchain",)
    # plt.suptitle(f"Kernel thickness: {kernel_thickness}")

    # plot_MPS_ensembles_2D(original = index.isel(realizations=0)*np.nan,
    #                 simulation= index,
    #                 year = year,
    #                 start_month = start_month,
    #                 end_month = end_month,
    #                 suptitle = "Tchain",)

    ################         

    ### lets use another interpolation method as comparison
    simulations_lin=gapped_data.interpolate_na(dim="time", method="linear",
                                              fill_value="extrapolate",
                                              use_coordinate=True,
                                              max_gap = pd.Timedelta(days = 14)) 
    # simulations_lin_verti =  gapped_data.interpolate_na(dim="depth", method="linear")
    simulations_subdlin=subdaily_linear_interp(gapped_data)

    #Vertical interpolation 
    verti_array_list = []
    for t in gapped_data.time:
        # Handle NaNs in the input data by filling them with interpolation
        depth_data = gapped_data.depth.data
        time_data = gapped_data.sel(time=t).data
        if np.isnan(time_data).any():
            # Fill NaNs using linear interpolation
            valid_mask = ~np.isnan(time_data)
            if np.all(~valid_mask):
                time_data = time_data
            else:
                inter_func_fill = interp1d(depth_data[valid_mask], time_data[valid_mask], kind='linear', 
                                           fill_value="extrapolate")
                time_data = inter_func_fill(depth_data)
        
        # Create the interpolation function with extrapolation
        inter_func = interp1d(depth_data, time_data, kind='linear', fill_value="extrapolate")
        interp_data = inter_func(data_original.depth.data)
        verti_array_list.append(interp_data)
    
    verti_array = xr.DataArray(np.array(verti_array_list),
                            dims=["time", "depth"],
                            coords={"time": gapped_data.time, "depth": data_original.depth})
    sim_vinterp = verti_array.transpose("depth", "time")
    # plt.figure()
    # verti_array.plot(vmin = 0,vmax = 40)
    # plt.figure()
    # tchain.temp.plot(vmin = 0,vmax = 40)
    # plt.figure()
    # simulations.sel(realizations = 1).plot(vmin = 0,vmax = 40)
    # plt.figure()
    # simulations_lin.transpose("depth", "time").plot(vmin = 0,vmax = 40)

    # 2D interpolation
    from scipy import interpolate
    #interpolate gapped_data in 2D
    x = np.arange(gapped_data.shape[1])
    y = np.arange(gapped_data.shape[0])
    masked_data = np.ma.masked_invalid(gapped_data)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~masked_data.mask]
    y1 = yy[~masked_data.mask]
    newarr = masked_data[~masked_data.mask]
    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                (xx, yy),
                                method='linear')
    sim_2dinterp = xr.DataArray(GD1, dims=["depth", "time"],
                            coords={"time": gapped_data.time, "depth": data_original.depth})


    # check 1D variogram output between the two methods
    ### now this is obviously for 5% missing values and gap length 8

    depth_array = tchain.depth.data

    # Initialize lists to store the RMSE values for each depth
    member_mean_list = []
    member_median_list = []
    lin_mean_list = []
    subdlin_mean_list = []
    lin2d_mean_list = []
    verti_mean_list = []

    for single_depth in depth_array:
        print("Looking at depth: " + str(single_depth))
        qs_mean = simulations.sel(depth=single_depth, method="nearest").mean(dim="realizations").squeeze()
        qs_median = simulations.sel(depth=single_depth, method="nearest").median(dim="realizations").squeeze()
        print("load the mean and median data")
        qs_mean = qs_mean.load()
        qs_median = qs_median.load()

        sims = np.array([qs_mean, qs_median,
                        simulations_lin.sel(depth=single_depth, method="nearest").squeeze(),
                        simulations_subdlin.sel(depth=single_depth, method="nearest").squeeze(),
                        sim_2dinterp.sel(depth = single_depth,method = 'nearest'),
                        sim_vinterp.sel(depth = single_depth,method = 'nearest')],)
                        # simulations_lin_verti.sel(depth=single_depth, method="nearest").squeeze())
        bin_corrector = 24 / obs_in_day
        print(sims.shape)

        print("now computing variogram")
        print(datetime.datetime.now())
        bin_centers, gamma_obs, gamma_sim_list = compare_variograms_nothreads(data_original.sel(depth = single_depth),
                                                                            sims,
                                                                            gap_indices=None,
                                                                            bin_number=int(96 / int(bin_corrector)))
        
        
        # plot_variograms(bin_centers, gamma_obs, gamma_sim_list, varname, single_depth)


        rmse_var_list = []
        for sim in range(sims.shape[0]):
            rmse = np.round(np.sqrt(np.nanmean((gamma_obs - gamma_sim_list[sim])**2)), 4)
            rmse_var_list.append(rmse)
        
        member_mean_list.append(rmse_var_list[0])
        member_median_list.append(rmse_var_list[1])
        lin_mean_list.append(rmse_var_list[2])
        subdlin_mean_list.append(rmse_var_list[3])
        lin2d_mean_list.append(rmse_var_list[4])
        verti_mean_list.append(rmse_var_list[5])

        print(rmse_var_list[0])
        print(rmse_var_list[1])
        print(rmse_var_list[2])
        print(rmse_var_list[3])
        print(rmse_var_list[4])
        print(rmse_var_list[5])

    sim_mean_rmses[kernel_thickness] = member_mean_list
    sim_median_rmses[kernel_thickness] = member_median_list    


#%%
mask = [23,24,26,27]
new_depth = np.delete(depth_array,mask)
new_lin_mean_list = np.delete(lin_mean_list,mask)
new_subdlin_mean_list = np.delete(subdlin_mean_list,mask)
new_lin2d_mean_list = np.delete(lin2d_mean_list,mask)
new_verti_mean_list = np.delete(verti_mean_list,mask)

print(f"Horizontal Linear: mean RMSE=", new_lin_mean_list.mean())
print(f"Horizontal Subdaily linear: mean RMSE=", new_subdlin_mean_list.mean())
print(f"2D linear: mean RMSE=", new_lin2d_mean_list.mean())
print(f"Vertical linear: mean RMSE=", new_verti_mean_list.mean())


# Plotting the results
plt.figure(figsize=(6, 12))
blues = sns.color_palette("Blues", len(kernel_thicknesses))
reds = sns.color_palette("Reds", len(kernel_thicknesses))
for k in kernel_thicknesses:
    new_sim_median_rmses = np.delete(sim_median_rmses[k],mask)
    print(f"Kernel thickness {k} : mean RMSE=", new_sim_median_rmses.mean())
    plt.plot(new_sim_median_rmses,new_depth, label=f"{k} layer kernel", color=reds[kernel_thicknesses.index(k)])
    # plt.plot(sim_mean_rmses[k],depth_array,   color=blues[kernel_thicknesses.index(k)])
    # plt.plot(sim_median_rmses[k],depth_array, label=f"{k} layer kernel", color=reds[kernel_thicknesses.index(k)])



plt.plot(new_lin_mean_list, new_depth, label='Linear', color = 'black',linestyle = '--')
plt.plot(new_subdlin_mean_list, new_depth, label='Subdaily linear', color = 'black',linestyle = '-')
plt.plot(new_lin2d_mean_list, new_depth, label='2D linear', color = 'black',linestyle = '-.')
plt.plot(new_verti_mean_list, new_depth, label='Vertical linear', color = 'black',linestyle = ':')


# plt.plot(member_mean_list, depth_array, label='Member Mean')
# plt.plot(member_median_list, depth_array, label='Member Median')
# plt.plot(lin_mean_list[~mask], depth_array[~mask], label='Linear', color = 'black',linestyle = '--')
# plt.plot(subdlin_mean_list, depth_array, label='Subdaily linear', color = 'black',linestyle = '-')
# plt.plot(lin2d_mean_list, depth_array, label='2D linear', color = 'black',linestyle = '-.')
# plt.plot(verti_mean_list, depth_array, label='Vertical linear', color = 'black',linestyle = ':')

plt.ylim(0,-50)
plt.xlabel('Variogram RMSE')
plt.ylabel('Depth')
plt.title('Variogram RMSE vs Depth')
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to have depth increasing downwards
plt.grid()
plt.savefig("/home/pwiersma/scratch/Figures/Lexplore/Tchain_variogram_rmse_plus2dverti.png", dpi = 300)

#%% Mean rmse 
for k in kernel_thicknesses:
    print()


#%% check 2D variogram output between the two methods
# s = simulations.isel(realizations=0)
# gs.vario_estimate(s)
# %%

f1,axes = plt.subplots(3,2,figsize=(15,15))
axes = axes.flatten()

tchain.temp.plot(ax = axes[0],vmin = 0,vmax = 40)
gapped_data.plot(ax = axes[1],vmin = 0,vmax = 40)
simulations.sel(realizations = 1).plot(ax = axes[2],vmin = 0,vmax = 40)
sim_2dinterp.plot(ax = axes[3],vmin = 0,vmax = 40)
simulations_lin.transpose("depth", "time").plot(ax = axes[4],vmin = 0,vmax = 40)
sim_vinterp.plot(ax = axes[5],vmin = 0,vmax = 40)

axes[0].set_title("Original")
axes[1].set_title("Original + artifiacial gaps")
axes[2].set_title("QS - 10 kernels")
axes[3].set_title("2D linear interpolation")
axes[4].set_title("Linear horizontal")
axes[5].set_title("Linear vertical")

plt.grid()



# %% horizontal plots for depths d
depths = [-0.5,-1,-20,-21,-48]
f1, axes = plt.subplots(len(depths),1,figsize=(10,10))
for i,d in enumerate(depths):
    tchain.temp.sel(depth = d, method = 'nearest').plot(ax = axes[i],
                        color = 'tab:blue',label = 'Original',zorder = 100)
    simulations.sel(realizations = 1,depth = d, method = 'nearest').plot(ax = axes[i],
            color = 'tab:red',label = 'QS - 10 kernels',zorder = 90)  
    sim_2dinterp.sel(depth = d, method = 'nearest').plot(ax = axes[i],
                        color = 'tab:green',label = '2D linear interpolation',zorder = 80)
    simulations_lin.sel(depth = d, method = 'nearest').plot(ax = axes[i],
                    color = 'tab:orange',label = 'Linear horizontal')
    simulations_subdlin.sel(depth = d, method = 'nearest').plot(ax = axes[i],
                    color = 'tab:grey',label = 'Subdaily linear')
    sim_vinterp.sel(depth = d, method = 'nearest').plot(ax = axes[i],
                    color = 'tab:purple',label = 'Linear vertical')
    axes[i].set_title(f"Depth: {d}")
    axes[i].legend()
    axes[i].grid()
# %% calculate rmse between tchain.temp and the different simulations
rmse_qs = np.sqrt(np.nanmean((tchain.temp - simulations.sel(realizations = 1))**2))
rmse_2d = np.sqrt(np.nanmean((tchain.temp - sim_2dinterp)**2))
rmse_horiz = np.sqrt(np.nanmean((tchain.temp - simulations_lin)**2))
rmse_subdlin = np.sqrt(np.nanmean((tchain.temp - simulations_subdlin)**2))
rmse_verti = np.sqrt(np.nanmean((tchain.temp - sim_vinterp)**2))

print(f"RMSE QS: {rmse_qs}")
print(f"RMSE 2D: {rmse_2d}")
print(f"RMSE Horiz: {rmse_horiz}")
print(f"RMSE Subdlin: {rmse_subdlin}")
print(f"RMSE Verti: {rmse_verti}")


# %%
