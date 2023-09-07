#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
import pandas as pd
import netCDF4
import os
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
from g2s import g2s
from random import randrange


# In[ ]:


def compute_gap_sizes(var):
    """Returns a list containing the size of all gaps in var
    var should be a 1D xarray.DataArray"""
    gap_sizes = []
    j = 0
    for i in range(var.data.size):
        if i<j+1: continue
        if np.isnan(var.data[i]):
            count = 1
            if i == var.data.size-1: #Last data entry
                gap_sizes.append(count)
                break
            else:
                j=i
                while np.isnan(var.data[j+1]):
                    count+=1
                    j+=1
                    if j==var.data.size-1:
                        break
                gap_sizes.append(count)
    return np.array(gap_sizes)

def gap_info(var,varname,plot_folder):
    print(f"{varname} has {np.count_nonzero(np.isnan(var.data))} nans in {var.data.size} data points")
    gap_sizes = compute_gap_sizes(var)
    print(f"{varname} has {np.round(np.count_nonzero(np.isnan(var.data))/var.data.size*100,2)}% missing values")
    print(f"{varname} has {len(gap_sizes)} gaps with {np.median(gap_sizes)} median gap size")
    print(f"{varname} has {len(gap_sizes)} gaps with {np.round(np.mean(gap_sizes),2)} mean gap size")
    
    plt.figure(figsize=(10,4))
    plt.title(f"{np.count_nonzero(np.isnan(var.data))} nans, {len(gap_sizes)} gaps, {np.median(gap_sizes)} median gap size, {np.round(np.mean(gap_sizes),2)} mean gap size, {np.round(np.count_nonzero(np.isnan(var.data))/var.data.size*100,2)}% missing values")
    plt.boxplot(np.array(gap_sizes),vert = False)
    plt.tight_layout()
    plt.savefig(plot_folder+varname+"_gap_distribution.png")
    plt.show()
    return
    
def create_gaps(var, gap_number,max_gap_size = 14):
    """Creates new gaps in var with the by resampling from existing gap sizes 
    Returns var with gaps and a list with all indices of the gap locations"""
    new_var = var.copy()
    gap_sizes = compute_gap_sizes(new_var) #number and size of gaps
    gap_sizes[gap_sizes>max_gap_size] = max_gap_size
    #gap_distribution = DiscreteAliasUrn(gap_sizes) #distribution derived from gap_sizes, this didn't always work
    #new_gaps = gap_distribution.rvs(gap_number) #new realizations of this distribution
    new_gaps = random.choices(sorted(list(gap_sizes)),k=gap_number)
    gap_indices = []
    for gap in new_gaps: #Try to find space for each new gap in the original data
        filled = False
        loop = 0
        while filled==False:
            if loop == 10000:
                print(f"Gap with size {gap} is too large")
                break
            rnd_idx = np.random.randint(0,len(new_var.data))
            if rnd_idx + gap >= len(new_var.data): # Out of bounds
                continue
            if np.count_nonzero(np.isnan(new_var.data[rnd_idx:rnd_idx+gap]))!=0: #Data already has gaps here
                continue
            timerange = slice(new_var.time[rnd_idx],new_var.time[rnd_idx+gap-1])
            new_var.loc[dict(time = timerange)] = np.nan
            for j in range(0,gap): 
                gap_indices.append(rnd_idx+j)
            filled = True
    return new_var, gap_indices

#Calendar day representation using a sinus and cosinus function with wavelength 365.25 days
def sin_costfunction(length,daily_timesteps = 4): #With 4 timesteps per day the cost function needs to be elongated 4 times to fit in a year
    x= np.linspace(0,length-1,length)
    return  np.sin(x*(np.pi*2)/(365.25*daily_timesteps))
def cos_costfunction(length,daily_timesteps = 4):
    x= np.linspace(0,length-1,length)
    return  np.cos(x*(np.pi*2)/(365.25*daily_timesteps))

def rmse(original, simulation):
    return np.sqrt(np.nanmean((original-simulation)**2))

def plot_MPS_ensembles(original, simulation, year, start_month, end_month,plot_folder, alpha = 0.5,title=None):
    """Plots the original data and an ensemble of simulations in a single figure for several months"""
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulation.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = alpha,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'tab:blue',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} QS ensembles","Original data"])
    ax1.grid()
    RMSE = rmse(original.data,simulation.data)
    ax1.set_title( f"{title} RMSE = {np.round(RMSE,3)}")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_simulation_example.png")
    plt.savefig(plot_folder+title+"_simulation_example.pdf")
    plt.show()
    return
    
    

def normalize_image(array):
    """Normalizes an array 
    sklearn.preprocessing can only process 2D arrays
    array should have the number of variables as the last dimension
    Returns tuple of normalized input and the applied scaler"""
    flag_3D = False
    if array.ndim == 3:
        flag_3D = True
        dim1,dim2,dim3 = array.shape
        array = np.reshape(array,(dim1*dim2,dim3))
    
    standard_scaler = StandardScaler()
    norm_array = standard_scaler.fit_transform(array)
    
    if flag_3D == True:
        norm_array = np.reshape(norm_array,(dim1,dim2,dim3))
    return norm_array,standard_scaler

def undo_normalize_image(norm_array,scaler):
    flag_3D = False
    if norm_array.ndim ==3:
        flag_3D = True
        dim1,dim2,dim3 = norm_array.shape
        norm_array = np.reshape(norm_array,(dim1*dim2,dim3))
    
    standard_scaler = scaler
    array = standard_scaler.inverse_transform(norm_array)
    
    if flag_3D==True:
        array = np.reshape(array,(dim1,dim2,dim3))
    return array

def ensemble_QS(N,**args):
    """Inputs: no. of ensemble runs N and QS arguments
    Mandatory and optional parameters same as in QS"""
    simulation_list = []
    args={k: v for k, v in args.items() if v is not None}
    
    if args['ti'].ndim !=1:
        args['ti'],ti_scaler = normalize_image(args['ti'])
        args['di'],di_scaler = normalize_image(args['di'])
    
    for i in range(N):
        simulation,index,time,progress,jobid = g2s(a='qs',**args);
        if args['ti'].ndim!=1:
            simulation = undo_normalize_image(simulation,di_scaler)
        simulation_list.append(simulation)
    simulations_stack = np.stack(simulation_list)
    
    return simulations_stack

def unify_time_axis(da1,da2):
    if np.max(da1.time.values)>np.max(da2.time.values):
        end=da2.time.values[-1]
    else:
        end=da1.time.values[-1]
    if np.min(da1.time.values)<np.min(da2.time.values):
        start=da2.time.values[0]
    else:
        start=da1.time.values[0]
    da1_cut=da1.sel(time=slice(start,end))
    da2_cut=da2.sel(time=slice(start,end))
    return da1_cut,da2_cut

def create_gap_index(da,gap_percent,gap_length):
    """Inputs: data array, how many percent of missing data from the data array length we wanna create, length of each gap 
    Output: A list of random locations where to put the gaps.""" 
    len_var=da.data.size
    gap_amount_num=int(np.round(gap_percent/100*len_var))
    gap_number=int(np.round(gap_amount_num/gap_length))
    gap_location=[]
    for i in range(gap_number):
        rand_location=randrange(len_var)
        # the 50 is a bit arbitraty, its like 96/2 and some, while 96 being probably our biggest gap we check for
        # the 48 left and right are also a bit arbitrary, it is basically the biggest gap I wanna check (96)
        while np.sum(np.isnan(da[rand_location-48:rand_location+48].values))>0 or rand_location <50 or rand_location >len_var-50:
            rand_location=randrange(len_var)
        gap_location.append(rand_location)
    return gap_location

def create_gapped_ts(da,gap_locations,gap_length,selector=1):
    """This one introduces nans at the gap locations for a certain length""" 
    # the selector has to be 1 for 24, 2 for 48,3 for 72, 4 for 96 etc
    print("Amount NAs in orig :"+str(np.isnan(da.values).sum()))
    
    len_var=da.data.size
    print("% NAs in orig :"+str(np.isnan(da.values).sum()/len_var*100))
    da_new=da.copy()
    if selector>1:
        gap_locations=gap_locations[0::selector]
    for i in gap_locations:
        da_new[int(i-gap_length/2):int(i+gap_length/2)]=np.nan
    print("Amount NAs in new :"+str(np.isnan(da_new.values).sum()))
    print("% NAs in new :"+str(np.isnan(da_new.values).sum()/len_var*100))
    print("Added % NAs :"+str((np.isnan(da_new.values).sum()-np.isnan(da.values).sum())/len_var*100))
    return da_new

def univ_g2s(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    for run in runs:
        for percent in percent_list:
            gap_locations,ds24=create_gap_index_nooverlap(da=data_original,gap_percent=percent,gap_length=24)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts(da=data_original,gap_locations=gap_locations,gap_length=gap_amount_list[i],selector=selector_list[i])
                L = gapped_data.data.size
                sin_calendar = sin_costfunction(L,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(L,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))

                #Univariate gap-filling
                name_addedinfo="UV"
                ti = gapped_data.data
                di = gapped_data.data
                dt = [0]

                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked,coords = {'realizations':np.arange(1,stacked.shape[0]+1),'time':gapped_data.time})

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,error,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)
                
                df.to_csv(output_name, index=False)
    return simulations,df

def day_of_year_g2s(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    for run in runs:
        for percent in percent_list:
            gap_locations,ds24=create_gap_index_nooverlap(da=data_original,gap_percent=percent,gap_length=24)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts(da=data_original,gap_locations=gap_locations,gap_length=gap_amount_list[i],selector=selector_list[i])
                L = gapped_data.data.size
                sin_calendar = sin_costfunction(L,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(L,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))

                #Univariate gap-filling
                name_addedinfo="calday"
                ti = np.stack([gapped_data.data,sin_calendar,cos_calendar],axis = 1)
                di = np.stack([gapped_data, sin_calendar,cos_calendar],axis = 1)
                dt = [0,0,0] #3 continuous variables

                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,0], 
                                            coords = {'realizations':np.arange(1,stacked.shape[0]+1),'time':gapped_data.time})

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,error,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)

                df.to_csv(output_name, index=False)
    return simulations,df


def time_of_day_of_year_g2s(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    for run in runs:
        for percent in percent_list:
            gap_locations,ds24=create_gap_index_nooverlap(da=data_original,gap_percent=percent,gap_length=24)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts(da=data_original,gap_locations=gap_locations,gap_length=gap_amount_list[i],selector=selector_list[i])
                L = gapped_data.data.size
                sin_calendar = sin_costfunction(L,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(L,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))

                #Univariate gap-filling
                name_addedinfo="caldaytimeday"
                ti = np.stack([gapped_data.data,
                            sin_calendar,
                            cos_calendar,
                            timeofday],axis = 1)
                di = np.stack([ gapped_data.data,
                            sin_calendar,
                            cos_calendar,
                            timeofday],axis = 1)
                #ki = np.ones([L,4])
                #ki[:,:3] = 0.5 #Assign half weight to categorical variable 
                dt = [0,0,0,1]  #time of day is a categorical variable

                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)

                simulations = xr.DataArray(data =stacked[:,:,0],
                                            coords = {'realizations':np.arange(1,stacked.shape[0]+1),'time':gapped_data.time})        

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,error,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)

                df.to_csv(output_name, index=False)
    return simulations,df

def one_cov_g2s(original,var1,cov,var2,cov_name,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name):
    data_original = original[var1]
    output_name=csv_folder+name+var1+".csv"
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    for run in runs:
        for percent in percent_list:
            gap_locations,ds24=create_gap_index_nooverlap(da=data_original,gap_percent=percent,gap_length=24)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts(da=data_original,gap_locations=gap_locations,gap_length=gap_amount_list[i],selector=selector_list[i])
                L = gapped_data.data.size
                sin_calendar = sin_costfunction(L,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(L,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))

                #gap-filling with one covariate
                covar2 = cov[var2].copy()
                name_addedinfo=cov_name
                #covar2.loc[dict(time = covar2.time[gap_indices])] = np.nan

                ti = np.stack([gapped_data.data,
                            covar2],axis = 1)
                di = np.stack([gapped_data.data,
                            covar2],axis = 1)
                dt = [0,0,] 
                #ki = np.ones([L,5])
                #ki[:,:4] = 0.3 #Assign half weight to categorical variable 


                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,0],
                                            coords = {'realizations':np.arange(1,stacked.shape[0]+1),'time':gapped_data.time})

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,error,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)

                df.to_csv(output_name, index=False)
    return simulations,df


def create_gap_index_test(da,gap_percent,gap_length):
    """Inputs: data array, how many percent of missing data from the data array length we wanna create, length of each gap 
    Output: A list of random locations where to put the gaps.""" 
    len_var=da.data.size
    gap_amount_num=int(np.round(gap_percent/100*len_var))
    gap_number=int(np.round(gap_amount_num/gap_length))
    gap_location=[]
    for i in range(gap_number):
        rand_location=randrange(len_var)
        empty_list=[0]
        # the 50 is a bit arbitraty, its like 96/2 and some, while 96 being probably our biggest gap we check for
        # the 48 left and right are also a bit arbitrary, it is basically the biggest gap I wanna check (96)
        while np.sum(empty_list)>0 or np.sum(np.isnan(da[rand_location-48:rand_location+48].values))>0 or rand_location <50 or rand_location >len_var-50:
            rand_location=randrange(len_var)
            empty_list=[0]
            for i in range(len(gap_location)):
                lower_limit=np.asarray(gap_location)-50
                upper_limit=np.asarray(gap_location)+50
                empty_list.append(lower_limit[i] <= rand_location <= upper_limit[i])
            
            
            
            
        gap_location.append(rand_location)
    return gap_location

def create_gap_index_nooverlap(da,gap_percent,gap_length):
    """Inputs: data array, how many percent of missing data from the data array length we wanna create, length of each gap 
    Output: A list of random locations where to put the gaps.""" 
    len_var=da.data.size
    da_cp=da.copy()
    gap_amount_num=int(np.round(gap_percent/100*len_var))
    gap_number=int(np.round(gap_amount_num/gap_length))
    gap_location=[]
    for i in range(gap_number):
        rand_location=randrange(len_var)
        
        # the 50 is a bit arbitraty, its like 96/2 and some, while 96 being probably our biggest gap we check for
        # the 48 left and right are also a bit arbitrary, it is basically the biggest gap I wanna check (96)
        while np.sum(np.isnan(da_cp[rand_location-51:rand_location+51].values))>0 or rand_location <51 or rand_location >len_var-51:
            
            rand_location=randrange(len_var)
        gap_location.append(rand_location)
        da_cp[rand_location-int(gap_length/2):rand_location+int(gap_length/2)]=np.nan
    return gap_location,da_cp

