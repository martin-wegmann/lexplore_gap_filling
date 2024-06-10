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


# ### Thetis

# In[9]:


thetis=xr.open_dataset(input_folder+"thetis_3hr_g2s.nc")


# ### Idronaut

# In[11]:


idronaut=xr.open_dataset(input_folder+"idronaut_3hr_g2s.nc")


# ### Lake Reanalysis

# In[13]:


meteolakes=xr.open_dataset(input_folder+"meteolakes_g2s.nc")


# In[14]:


meteolakes=meteolakes.sel(depth=tchain.depth.values,method="nearest")


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

# In[20]:


varname="temp"


# In[22]:


gap_info(var=tchain[varname],varname="tchain_temp",plot_folder=plots_folder)


# ### Run gapfilling

# In[23]:



# #### eval phase

# In[ ]:





# In[64]:





# In[ ]:





# In[ ]:


#tchain, linear depth pen
#tchain, inverse depth pen
#tchain, var pen
#tchain, best depth penal, yearly cycle
#tchain, best depth penal, yearly cycle, daily cycle
#tchain, best depth penal, model
#tchain, best depth penal, thetis
#tchain, best depth penal, idronaut
#tchain, best depth penal, best1, best2,


# In[ ]:





# In[24]:


percent_list=[5]
gap_amount_list=[8,16,24]
selector_list=[1,2,3]
N = 25
test_runs=10

columns_df=["NAME",'RUN',"MEMBER","PERC","GAP_SIZE","CORR","CORR_lin","CORR_akima","CORR_spline","CORR_quad","CORR_pchip","CORR_subdlin","RMSE","RMSE_lin","RMSE_akima","RMSE_spline","RMSE_quad","RMSE_pchip","RMSE_subdlin","STDR"]

df = pd.DataFrame(columns=columns_df)


# In[25]:


#print(datetime.datetime.now())


# In[26]:


#filled_data,error_df=univ_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="linear")


# In[27]:


print(datetime.datetime.now())


# In[ ]:


#filled_data,error_df=univ_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="inverse")


# In[ ]:


print(datetime.datetime.now())


# In[ ]:


#filled_data,error_df=univ_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")


# In[ ]:


#print(datetime.datetime.now())

#filled_data,error_df=day_of_year_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")


#filled_data,error_df=time_of_day_of_year_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")

#print(datetime.datetime.now())

#filled_data,error_df=one_cov_g2s_2D(original=tchain,var1=varname,cov=meteolakes,var2="Temp",cov_name="model",obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,plot_folder=plots_folder,name="tchain_eval_",depan="var")

#filled_data,error_df=one_cov_g2s_2D(original=tchain,var1=varname,cov=thetis,var2="temp",cov_name="thetis",obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,plot_folder=plots_folder,name="tchain_eval_",depan="var")

#filled_data,error_df=one_cov_g2s_2D(original=tchain,var1=varname,cov=idronaut.transpose(),var2="temp",cov_name="idronaut",obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,plot_folder=plots_folder,name="tchain_eval_",depan="var")
#


# #### error matrix phase


percent_list=[5,10,15]
gap_amount_list=[8,16,24,32,40,48]
selector_list=[1,2,3,4,5,6]
N = 25
test_runs=5

columns_df=["NAME",'RUN',"MEMBER","PERC","GAP_SIZE","CORR","CORR_lin","CORR_akima","CORR_spline","CORR_quad","CORR_pchip","CORR_subdlin","RMSE","RMSE_lin","RMSE_akima","RMSE_spline","RMSE_quad","RMSE_pchip","RMSE_subdlin","RMSE_VAR_mean","RMSE_VAR_median","RMSE_VAR_lin","RMSE_VAR_akima","RMSE_VAR_spline","RMSE_VAR_quad","VAR_pchip","RMSE_VAR_subdlin","STDR","DEPTH"]

df = pd.DataFrame(columns=columns_df)


filled_data,error_df=day_of_year_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,plot_folder=plots_folder,name="tchain_season_",depan="var")
#

# #### reconstruction phase
N=50

gapped_data=tchain[varname].copy()
data_original=gapped_data.copy()

 timeofday = data_original.time.dt.hour.values #C

depth_dim, time_dim = data_original.shape

depth_linear = np.transpose(np.tile(data_original.depth.data,(time_dim,1)))
    
depth_inverse = 1/depth_linear

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
    
if da_var_depth.isnull().sum().values>15:
	depth_variance=np.flip(np.transpose(np.tile(np.log(np.arange(1,39,1)),(time_dim,1))))
    

sin_calendar = sin_costfunction(time_dim ,daily_timesteps = obs_in_day)
cos_calendar = cos_costfunction(time_dim ,daily_timesteps = obs_in_day)

sin_2D = np.tile(sin_calendar, (depth_dim,1))
cos_2D = np.tile(cos_calendar, (depth_dim,1))

ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
dt = [0,0,0,0]

L4_stacked = ensemble_QS(N = N,ti=ti, di=di,dt=dt, k=1.2,n=50,j=0.5,ki=None)

L4_simulation = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time}) 
                


