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


# #### eval phase




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


print(datetime.datetime.now())


# In[26]:
#tchain, linear depth pen


filled_data,error_df,indices=univ_g2s_2D(original=tchain,
                                 var=varname
                                 ,obs_in_day=timestepsinday,
                                 N=N,percent_list=percent_list,
                                 gap_amount_list=gap_amount_list,
                                 selector_list=selector_list,
                                 test_runs=test_runs,
                                 df=df,
                                 csv_folder=output_folder,
                                 name="tchain_eval_",
                                 depan="linear")


def plot_filled_vs_indices(filled_data,indices, depth, realization, error_threshold = 0.5):
    index = indices.sel(depth=depth, realizations=realization)
    index = xr.where(index == 0 , np.nan, index)

    error = (filled_data - tchain['temp']).sel(depth =   depth, realizations = realization)
    high_errormask = np.abs(error) >error_threshold
    
    filled = filled_data.sel(depth=depth, realizations=realization)
    orig = tchain['temp'].sel(depth=depth)

    high_error_indices = index.where(high_errormask)


    f, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10),sharex = True)
    filled.plot(ax = ax1, label = 'Filled')
    orig.plot(ax = ax1, label = 'Original')
    ax1.legend()
    ax1.set_title('Simulated vs Observed ')
    ax1.grid()

    #plot the index as points without lines
    index.plot(ax = ax2, marker = 'o', linestyle = 'none',label = 'Filled index')
    #Red crosses for the indices with high errors, for those we are most interested where they come from
    high_error_indices.plot(ax = ax2, marker = 'x', linestyle = 'none', color = 'red',
                            label = f"Error > {error_threshold}")
    ax2.set_ylabel('Original index of filled values')
    ax2.grid()
    ax2.legend()
    plt.show()

depth = -1.0
realization = 1

plot_filled_vs_indices(filled_data,
                       indices,
                         depth, realization)

# In[27]:


print(datetime.datetime.now())


# In[ ]:
#tchain, inverse depth pen

filled_data,error_df=univ_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="inverse")


# In[ ]:


print(datetime.datetime.now())


# In[ ]:
#tchain, var pen

filled_data,error_df=univ_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")


# In[ ]:
#tchain, best depth penal, yearly cycle

print(datetime.datetime.now())

filled_data,error_df=day_of_year_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")

#tchain, best depth penal, yearly cycle, daily cycle
filled_data,error_df=time_of_day_of_year_g2s_2D(original=tchain,var=varname,obs_in_day=timestepsinday,N=N,percent_list=percent_list,gap_amount_list=gap_amount_list,selector_list=selector_list,test_runs=test_runs,df=df,csv_folder=output_folder,name="tchain_eval_",depan="var")

print(datetime.datetime.now())

#tchain, best depth penal, model
#tchain, best depth penal, thetis
#tchain, best depth penal, idronaut
#tchain, best depth penal, best1, best2,

