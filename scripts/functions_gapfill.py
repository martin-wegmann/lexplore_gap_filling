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
from concurrent.futures import ThreadPoolExecutor
import gstools as gs
import statsmodels.api as sm



# In[ ]:


def compute_gap_sizes(var):
    """Returns a list containing the size of all gaps in var
    var should be a 1D xarray.DataArray"""
    if len(var.shape)==2:
        var=var.stack(z=("depth", "time"))
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
    
    if len(var.shape)==2:
        var=var.stack(z=("depth", "time"))
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
    
    
    simulations_lin=original.interpolate_na(dim="time", method="linear")
    simulations_slin=original.interpolate_na(dim="time", method="slinear")
    simulations_akima=original.interpolate_na(dim="time", method="akima")
    simulations_spline=original.interpolate_na(dim="time", method="spline")
    simulations_quad=original.interpolate_na(dim="time", method="quadratic")
    simulations_pchip=original.interpolate_na(dim="time", method="pchip")
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulation.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = alpha,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} QS ensembles","Original data"])
    ax1.grid()
    RMSE = rmse(original.data,simulation.data)
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_simulation_example.png")
    plt.savefig(plot_folder+title+"_simulation_example.pdf")
    plt.show()
    
    
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_lin.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} linear int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_linear_example.png")
    plt.savefig(plot_folder+title+"_linear_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_slin.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} slinear int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_slinear_example.png")
    plt.savefig(plot_folder+title+"_slinear_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_akima.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} akima int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_akima_example.png")
    plt.savefig(plot_folder+title+"_akima_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_spline.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} spline int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_spline_example.png")
    plt.savefig(plot_folder+title+"_spline_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_quad.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} quadratic int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_quadratic_example.png")
    plt.savefig(plot_folder+title+"_quadratic_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    ensemble = simulations_pchip.loc[dict(time=slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"))].plot.line(
        x= 'time',ax=ax1,alpha = 1,color='tab:red')
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} pchip int.","Original data"])
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_pchip_example.png")
    plt.savefig(plot_folder+title+"_pchip_example.pdf")
    plt.show()
    
    f1,ax1 = plt.subplots(figsize = (35,5))
    original_plot = original.loc[slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}")].plot(ax = ax1,color = 'black',label = 'Original',linewidth = 3)
    ax1.grid()
    ax1.set_title( f"{title} ")
    plt.tight_layout()
    plt.savefig(plot_folder+title+"_empty_example.png")
    plt.savefig(plot_folder+title+"_empty_example.pdf")
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
        while np.sum(np.isnan(da[rand_location-72:rand_location+72].values))>0 or rand_location <73 or rand_location >len_var-73:
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
            gap_locations,ds24=create_gap_index_nooverlap(da=data_original,gap_percent=percent,gap_length=obs_in_day)

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
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
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
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
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
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)

                df.to_csv(output_name, index=False)
    return simulations,df

def one_cov_g2s(original,var1,cov,var2,cov_name,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name,vario=False):
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
                #print(name_addedinfo)
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
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                if vario==True:
                    
                    qs_mean=simulations.mean(dim="realizations")
                    qs_median=simulations.median(dim="realizations")
                
                    sims = np.array([qs_mean,qs_median,simulations_lin,simulations_akima,simulations_spline,simulations_quad,simulations_pchip,simulations_subdlin])
                    bin_corrector=24/obs_in_day
                    
                    print("now computing variogram")
                    print(datetime.datetime.now())
                    bin_centers,gamma_obs, gamma_sim_list = compare_variograms(data_original,
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
                    akima_mean=rmse_var_list[3]
                    spline_mean=rmse_var_list[4]
                    quad_mean=rmse_var_list[5]
                    pchip_mean=rmse_var_list[6]
                    subdlin_mean=rmse_var_list[7]
                    
                    print("variogram done")
                    print(datetime.datetime.now())
                                             
                                             
                                       
                
                error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)
    
                if vario==True:
                    df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,member_mean,member_median,lin_mean,akima_mean,spline_mean,quad_mean,pchip_mean,subdlin_mean,std_ratio]], columns=df.columns)
                
                else:
                    df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
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
        while np.sum(empty_list)>0 or np.sum(np.isnan(da[rand_location-72:rand_location+72].values))>0 or rand_location <73 or rand_location >len_var-73:
            rand_location=randrange(len_var)
            empty_list=[0]
            for i in range(len(gap_location)):
                lower_limit=np.asarray(gap_location)-73
                upper_limit=np.asarray(gap_location)+73
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
        while np.sum(np.isnan(da_cp[rand_location-72:rand_location+72].values))>0 or rand_location <73 or rand_location >len_var-73:
            
            rand_location=randrange(len_var)
        gap_location.append(rand_location)
        da_cp[rand_location-int(gap_length/2):rand_location+int(gap_length/2)]=np.nan
    return gap_location,da_cp

def two_cov_g2s(original,var1,cov,var2,cov_name,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name):
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

                
                if len(var2)>1:
                    name_addedinfo=cov_name
                    #Create gaps at the same locations as AirTC
                    covar1 = cov[var2[0]].copy()
                    covar2 = cov[var2[1]].copy()

                    ti = np.stack([gapped_data.data,
                                covar1,
                                covar2],axis = 1)
                    di = np.stack([gapped_data.data,
                                covar1,
                                covar2],axis = 1)
                    dt = [0,0,0,] 
                    #ki = np.ones([L,5])
                    #ki[:,:4] = 0.3 #Assign half weight to categorical variable 
                    if len(var2)>2:
                        name_addedinfo=cov_name
                        #Create gaps at the same locations as AirTC
                        covar1 = cov[var2[0]].copy()
                        covar2 = cov[var2[1]].copy()
                        covar3 = cov[var2[2]].copy()

                        ti = np.stack([gapped_data.data,
                                    covar1,
                                    covar2,
                                    covar3],axis = 1)
                        di = np.stack([gapped_data.data,
                                    covar1,
                                    covar2,
                                    covar3],axis = 1)
                        dt = [0,0,0,0,] 
                        #ki = np.ones([L,5])
                        #ki[:,:4] = 0.3 #Assign half weight to categorical variable 
                else:
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
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)

                df.to_csv(output_name, index=False)
    return simulations,df


def exp_variogram(data,coords = None, gap_indices=None,  bin_number=None):
    """Calculates the experimental variogram of data using the gstools package. 
    For each distance bin the average squared difference is computed
    
    Parameters:
    ----------
    coords : coordinates of the data. If None simply take np.arange(data.size) as coordinates
    gap_indices: Provide in case you only want to calculate the variogram of the artificial gaps (to compare observations and simulations of these gaps)
    bin_number: number of bins to calculate the experimental variogram for. If None gstools provides a bin_number automatically
    
    
    """
    if not gap_indices==None:
        data_orig = data.copy()
        data = data*np.nan
        data[gap_indices] = data_orig[gap_indices]
    if coords ==None:
        coords = np.arange(data.size)
    if coords.size != data.size:
        print("Coords and data not equal in size")
        return
    if bin_number != None:
        bin_edges = np.arange(1,bin_number+1)
    else:
        bin_edges = None
    bin_center, gamma = gs.vario_estimate(coords,data,bin_edges)
    return bin_center,gamma

def compare_variograms(obs, simlist, decompose_seasonal =False, gap_indices=None, bin_number = None):
    """
    Compare the experimental variogram of observed data to the variograms of a list of simulated data.
    ThreadPoolExecutor() is used to parallelize the experimental variogram computatation. 

    Parameters:
    ----------
    obs : numpy array
        Time series of observed data.

    simlist : list of numpy arrays
        List of time series containing simulated data for comparison.

    decompose_seasonal : bool, optional (default=False)
        If True, decompose the observed data into seasonal trend and residual components before comparing variograms.
        Only the residuals are kept for calculation of the variograms. 

    gap_indices : list of int or None, optional (default=None)
        List of indices corresponding to artifical gaps. If provided, the variograms will only be calculated for these gaps. 

    bin_number : int or None, optional (default=None)
        Number of bins for variogram calculation. If None, an appropriate number of bins will be determined automatically.

    """
    if decompose_seasonal==True:
        if np.count_nonzero(np.isnan(obs))!=0: #decompose function doesn't take nans, so interpolate now and place back the nans later
            gaps = np.isnan(obs)
            obs = obs.interpolate('linear')
        obs_decomposed = sm.tsa.seasonal_decompose(obs, model = 'additive', period = 365)
        trend = obs_decomposed.trend
        obs = obs - trend
        if np.count_nonzero(np.isnan(obs))!=0:
            obs[gaps] = np.nan 
        simlist = [sim - trend for sim in simlist] #to make sure they are detrended in the exact same way
        
    with ThreadPoolExecutor() as executor:
        # Calculate the experimental variogram for obs in parallel
        print("future obs")
        future_obs = executor.submit(exp_variogram, obs, gap_indices=gap_indices, bin_number = bin_number)
        
        # Calculate the experimental variograms for simlist in parallel
        print("future sims")
        futures_sim = [executor.submit(exp_variogram, sim, gap_indices = gap_indices, bin_number = bin_number) for sim in simlist]
        
        # Retrieve results when ready
        print("retrieve future obs")
        bin_center, gamma_obs = future_obs.result()
        gamma_sim_list = []
        for future in futures_sim:
        	print("retrieve future sims")
        	gamma_sim_list.append(future.result()[1])
    return bin_center, gamma_obs, gamma_sim_list

def plot_variograms(bin_center, gamma_obs,gamma_sim_list,title = None,
                   xlabel = None, ylabel = None):
    f1,ax1 = plt.subplots(figsize = (15,7))
    ax1.scatter(bin_center, gamma_obs, color = 'black',marker = 'x',label ='Obs')
    for i,gamma_sim in enumerate(gamma_sim_list):
        ax1.scatter(bin_center,gamma_sim,label = f'Sim {i+1}')
        
    if title ==None:
        title = 'Experimental variogram comparison'
    else:
        title = title
    ax1.set_title(title)
    ax1.set_ylim(bottom = 0)
    ax1.set_ylabel(r"$\gamma$")
    ax1.set_xlabel('Lag')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
def rm_point(files):
    f=[]
    for i in range(len(files)):
        if not files[i].startswith("."):
            f.append(files[i])
    return f

    
def subdaily_linear_interp(data_array,times_of_day = 4):
    """
    Perform gap-filling linear interpolation between each of the subdaily time periods in the input data array.

    Parameters:
        data_array (xarray.DataArray): The input data array to be interpolated containing gaps 
        times_of_day (int) : The number of subdaily time periods in the input data array.

    Returns:
        xarray.DataArray: The interpolated data array with subdaily values.

    This function divides the input data array into times_of_day subsets, where each subset represents
    one of the four subdaily time periods. It then performs linear interpolation within each
    subset along the 'time' dimension and concatenates the results into a single data array.
    The final data array is sorted by the 'time' dimension.

    """
    l = []
    for t in range(times_of_day):
        d = data_array.isel(time=slice(t, None, times_of_day))
        l.append(d.interpolate_na(dim='time', method='linear'))
    subdaily_interpolation = xr.concat(l, dim='time').sortby('time')
    return subdaily_interpolation

def find_large_nan_gaps(arr, N):
    # Find indices where arr is not np.nan
    nan_indices = np.where(~np.isnan(arr))[0]
    # Calculate the gaps between NaNs
    nan_gaps = np.diff(nan_indices) - 1
    # Find indices where nan_gaps are larger than N
    large_gap_indices = np.where(nan_gaps > N)[0]
    # Initialize a list to store all the indices within large gaps
    all_gap_indices = []
    for gap_index in large_gap_indices:
        gap_start = nan_indices[gap_index] + 1  
        gap_end = nan_indices[gap_index + 1]    
        gap_indices = np.arange(gap_start, gap_end)
        all_gap_indices.extend(gap_indices.tolist())
    return all_gap_indices  

def generate_simulation_path_wo_gaps(di,max_gap_size):
    #Identify location of the large gaps
    large_gap_indices = find_large_nan_gaps(di,max_gap_size)
    #Start with a linear simulation path 
    indices = np.arange(di.size,dtype = float)
    #Take out the large gaps, -inf in the simulation path are skipped (np.nans are automatically filled)
    #indices[large_gap_indices] = -np.inf # Somehow this doesn't work 
    #Shuffle around the indices to create a random path first with and then without the large gaps
    sp = np.random.permutation(indices)
    sp[large_gap_indices] = -np.inf
    return sp

def create_gapped_ts_2D(da,gap_locations,depth_level_index,gap_length,selector=1):
    """This one introduces nans at the gap locations for a certain length""" 
    # the selector has to be 1 for 24, 2 for 48,3 for 72, 4 for 96 etc
    print("Amount NAs in orig :"+str(np.isnan(da.values).sum()))
    
    len_var=da.data.size
    print("% NAs in orig :"+str(np.isnan(da.values).sum()/len_var*100))
    da_new=da.copy()
    if selector>1:
        gap_locations=gap_locations[0::selector]
        depth_level_index=depth_level_index[0::selector]
    for i in range(len(gap_locations)):
        gap_location=gap_locations[i]
        depth_level=depth_level_index[i]
        da_new.isel(depth=depth_level)[int(gap_location-gap_length/2):int(gap_location+gap_length/2)]=np.nan
    print("Amount NAs in new :"+str(np.isnan(da_new.values).sum()))
    print("% NAs in new :"+str(np.isnan(da_new.values).sum()/len_var*100))
    print("Added % NAs :"+str((np.isnan(da_new.values).sum()-np.isnan(da.values).sum())/len_var*100))
    return da_new

def create_gap_index_nooverlap_2D(da,gap_percent,gap_length):
    """Inputs: data array, how many percent of missing data from the data array length we wanna create, length of each gap 
    Output: A list of random locations where to put the gaps.""" 
    len_var=da.data.size
    da_cp=da.copy()
    nrows=da.data.shape[0]
    ncols=da.data.shape[1]
    gap_amount_num=int(np.round(gap_percent/100*len_var)) # aber hier muss len_var schon das bleiben
    print("gap amount "+str(gap_amount_num))
    gap_number=int(np.round(gap_amount_num/gap_length))
    print("gap number "+str(gap_number))
    gap_location=[]
    depth_level_index=[]
    for i in range(gap_number):
       # print("loop")
        print("gap number currently finding "+str(i))

        rand_row=randrange(nrows)
        # print("row")
        #print(rand_row)

        da_cp_onelev=da_cp.isel(depth=rand_row)
        #print("cols")
        rand_location=randrange(ncols)
        
        #print(rand_location)
        # 24 because the max gap we look at is 48 tine steps. if it would be more than 48 time steps, this would need to be changed
        # this code crashes if it does not find suitable gaps
        while np.sum(np.isnan(da_cp_onelev[rand_location-24:rand_location+24].values))>0 or rand_location <24 or rand_location >ncols-24:

            rand_row=randrange(nrows)
            print("looking for suitable gaps")
            da_cp_onelev=da_cp.isel(depth=rand_row)
            rand_location=randrange(ncols)
        gap_location.append(rand_location)
        depth_level_index.append(rand_row)
        da_cp.isel(depth=rand_row)[rand_location-int(gap_length/2):rand_location+int(gap_length/2)]=np.nan
    return gap_location,depth_level_index,da_cp

def plot_MPS_ensembles_2D(original, simulation, year, start_month, end_month, alpha = 0.5, suptitle = None):
    f1,axes = plt.subplots(original.depth.size,1,figsize = (15,20),sharex = True)
    plt.subplots_adjust(hspace = 0.3)
    RMSE = rmse(original.data,simulation.data)
    plt.suptitle( f"{suptitle}, RMSE = {np.round(RMSE,3)}", y = 0.92, fontsize = 'x-large')
    for i,d in enumerate(original.depth.data):
        #original.loc[dict(time = slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"),
         #                   depth = d)].plot(ax = axes[i])
        ensemble = simulation.loc[dict(time = slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"),
                        depth = d)].plot.line(x = 'time',ax = axes[i], color = 'red', alpha = 0.3,add_legend=False)
        original_plot = original.loc[dict(time = slice(f"{str(year)}-{str(start_month)}",f"{str(year)}-{str(end_month)}"),
                        depth = d)].plot(ax = axes[i], color = 'blue',add_legend=False)
        axes[i].set_xlabel(None)
        axes[i].grid()
        partial_RMSE = rmse(original.sel({'depth':d}).data,simulation.sel({'depth':d}).data)
        axes[i].set_title( f"Depth:{d}, RMSE = {np.round(partial_RMSE,3)}")
    axes[0].legend([ensemble[0],original_plot[0]],[f"{len(ensemble)} QS ensembles","Original data"])



def univ_g2s_2D(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name,depan="linear"):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)

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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                
                sin_calendar = sin_costfunction(time_dim,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))

                #Univariate gap-filling
                
                if depan=="linear":
                    name_addedinfo="UVl"
                    ti = np.stack([gapped_data.data, depth_linear],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear],axis = 2)
                if depan=="inverse":
                    name_addedinfo="UVi"
                    ti = np.stack([gapped_data.data, depth_inverse],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse],axis = 2)
                if depan=="var":
                    name_addedinfo="UVv"
                    ti = np.stack([gapped_data.data, depth_variance],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance],axis = 2)
                dt = [0,0]

            
                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time})
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2))),4)
                error_akima = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2))),4)
                error_spline = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2))),4)
                error_quad = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2))),4)
                error_pchip = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2))),4)
                error_subdlin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2))),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                error = np.round(np.nanmean(np.sqrt(np.nanmean((simulations.data-data_original.data)**2))),4)
                std_ratio=np.round(np.nanmean((data_original/simulations).mean(dim="realizations").mean(dim="time").values),4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)
                
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]

                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()


    return simulations,df


def day_of_year_g2s_2D(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name,depan="linear"):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
               
                sin_calendar = sin_costfunction(time_dim ,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim ,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))


                name_addedinfo="calday"
                sin_2D = np.tile(sin_calendar, (depth_dim,1))
                cos_2D = np.tile(cos_calendar, (depth_dim,1))
                timeofday_2D = np.tile(timeofday, (depth_dim,1))

                if depan=="linear":
                    name_addedinfo="caldayl"
                    ti = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                if depan=="inverse":
                    name_addedinfo="caldayi"
                    ti = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                if depan=="var":
                    name_addedinfo="caldayv"
                    ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                dt = [0,0,0,0]

                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)

                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time}) 
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2))),4)
                error_akima = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2))),4)
                error_spline = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2))),4)
                error_quad = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2))),4)
                error_pchip = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2))),4)
                error_subdlin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2))),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                error = np.round(np.nanmean(np.sqrt(np.nanmean((simulations.data-data_original.data)**2))),4)
                std_ratio=np.round(np.nanmean((data_original/simulations).mean(dim="realizations").mean(dim="time").values),4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)
                
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]

                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()


    return simulations,df


def time_of_day_of_year_g2s_2D(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,name,depan="linear"):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                
                sin_calendar = sin_costfunction(time_dim ,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim ,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))


                name_addedinfo="caldaytimeday"
                sin_2D = np.tile(sin_calendar, (depth_dim,1))
                cos_2D = np.tile(cos_calendar, (depth_dim,1))
                timeofday_2D = np.tile(timeofday, (depth_dim,1))

                if depan=="linear":
                    name_addedinfo="caldaytimedayl"
                    ti = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D,timeofday_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D,timeofday_2D],axis = 2)
                if depan=="inverse":
                    name_addedinfo="caldaytimedayi"
                    ti = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D,timeofday_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D,timeofday_2D],axis = 2)
                if depan=="var":
                    name_addedinfo="caldaytimedayv"
                    ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D,timeofday_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D,timeofday_2D],axis = 2)
                dt = [0,0,0,0,1]

                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)

                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time}) 
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2))),4)
                error_akima = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2))),4)
                error_spline = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2))),4)
                error_quad = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2))),4)
                error_pchip = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2))),4)
                error_subdlin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2))),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                error = np.round(np.nanmean(np.sqrt(np.nanmean((simulations.data-data_original.data)**2))),4)
                std_ratio=np.round(np.nanmean((data_original/simulations).mean(dim="realizations").mean(dim="time").values),4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)
                
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]

                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()


    return simulations,df

def one_cov_g2s_2D(original,var1,cov,var2,cov_name,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,plot_folder,name,depan="linear",vario=False):
    data_original = original[var1]
    output_name=csv_folder+name+var1+".csv"
    
    covar = cov[var2].transpose().copy()
    name_addedinfo=cov_name
    
    
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)

    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)

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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)

            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                L = gapped_data.data.size
                sin_calendar = sin_costfunction(time_dim,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))


                print(gapped_data.shape)
                print(covar.shape)
                #Univariate gap-filling
                
                if depan=="linear":
                    name_addedinfo=name_addedinfo+"l"
                    ti = np.stack([gapped_data.data, depth_linear, covar.data],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear, covar.data],axis = 2)
                if depan=="inverse":
                    name_addedinfo=name_addedinfo+"i"
                    ti = np.stack([gapped_data.data, depth_inverse, covar.data],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse, covar.data],axis = 2)
                if depan=="var":
                    name_addedinfo=name_addedinfo+"v"
                    ti = np.stack([gapped_data.data, depth_variance, covar.data],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance, covar.data],axis = 2)
                dt = [0,0,0]

            
                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time})
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                error_lin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2))),4)
                error_akima = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2))),4)
                error_spline = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2))),4)
                error_quad = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2))),4)
                error_pchip = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2))),4)
                error_subdlin = np.round(np.nanmean(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2))),4)
                
                corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)

                corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                error = np.round(np.nanmean(np.sqrt(np.nanmean((simulations.data-data_original.data)**2))),4)
                std_ratio=np.round(np.nanmean((data_original/simulations).mean(dim="realizations").mean(dim="time").values),4)

                df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                df = pd.concat([df, df_temp], axis=0)
                
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]

                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()


    return simulations,df
    
    

    
def one_cov_g2s_2D_test(original,var1,cov,var2,cov_name,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,plot_folder,name,depan="linear",vario=True):
    data_original = original[var1]
    output_name=csv_folder+name+var1+".csv"
    depth_array=[-1.2,-5,-10,-15,-30,-48]
    
    covar = cov[var2].transpose().copy()
    name_addedinfo=cov_name
    
    
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)
    
    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    depth_dim, time_dim = data_original.shape
    
    depth_linear = np.transpose(np.tile(data_original.depth.data,(time_dim,1)))
    
    depth_inverse = 1/depth_linear
    mask_var=data_original.isnull().sum(dim="time")>(data_original.data.shape[1]/2)
    # where do we have more than 50% nans
    
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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)
    
            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                
                sin_calendar = sin_costfunction(time_dim,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))
    
                #Univariate gap-filling
                
                if depan=="linear":
                    name_addedinfo="UVl"
                    ti = np.stack([gapped_data.data, depth_linear],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear],axis = 2)
                if depan=="inverse":
                    name_addedinfo="UVi"
                    ti = np.stack([gapped_data.data, depth_inverse],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse],axis = 2)
                if depan=="var":
                    name_addedinfo="UVv"
                    ti = np.stack([gapped_data.data, depth_variance],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance],axis = 2)
                dt = [0,0]
    
            
                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time})
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                
                
                if vario==True:
                    
                    for single_depth in depth_array:
                        print("Looking at depth: "+str(single_depth ))
                        qs_mean=simulations.sel(depth=single_depth,method="nearest").mean(dim="realizations").squeeze()
                        qs_median=simulations.sel(depth=single_depth,method="nearest").median(dim="realizations").squeeze()
                        print("load the mean and median data")
                        qs_mean=qs_mean.load()
                        qs_median=qs_median.load()
                    
                        sims = np.array([qs_mean,qs_median,simulations_lin.sel(depth=single_depth,method="nearest").squeeze(),simulations_akima.sel(depth=single_depth,method="nearest").squeeze(),simulations_spline.sel(depth=single_depth,method="nearest").squeeze(),simulations_quad.sel(depth=single_depth,method="nearest").squeeze(),simulations_pchip.sel(depth=single_depth,method="nearest").squeeze(),simulations_subdlin.sel(depth=single_depth,method="nearest").squeeze()])
                        bin_corrector=24/obs_in_day
                        print(sims.shape)
                        
                        print("now computing variogram")
                        print(datetime.datetime.now())
                        bin_centers,gamma_obs, gamma_sim_list = compare_variograms(data_original,
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
                        akima_mean=rmse_var_list[3]
                        spline_mean=rmse_var_list[4]
                        quad_mean=rmse_var_list[5]
                        pchip_mean=rmse_var_list[6]
                        subdlin_mean=rmse_var_list[7]
                        
                        print("variogram done")
                        print(datetime.datetime.now())
                                                 
                                                 
                                           
                    
                        error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                        error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                        error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                        error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                        error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                        error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                    
                        corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                        corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                        corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                        corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                        corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)
    
                        corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                        error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                        std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)
    
                        df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,member_mean,member_median,lin_mean,akima_mean,spline_mean,quad_mean,pchip_mean,subdlin_mean,std_ratio,single_depth]], columns=df.columns)
                        df = pd.concat([df, df_temp], axis=0)
                
                else:



                
                    error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                    error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                    error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                    error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                    error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                    error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                    corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").values,4)
                    corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").values,4)
                    corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").values,4)
                    corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").values,4)
                    corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").values,4)
                    
    
                    corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").values,4)
                    
                    error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                    
                    std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").values,4)
                    df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                    df = pd.concat([df, df_temp], axis=0)
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]
    
                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()
    
    
    return simulations,df


def day_of_year_g2s_2D_test(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,plot_folder,name,depan="linear",vario=True):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    depth_array=[-1.2,-5,-10,-15,-30,-48]
    
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)
    
    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    depth_dim, time_dim = data_original.shape
    
    depth_linear = np.transpose(np.tile(data_original.depth.data,(time_dim,1)))
    
    depth_inverse = 1/depth_linear
    mask_var=data_original.isnull().sum(dim="time")>(data_original.data.shape[1]/2)
    # where do we have more than 50% nans
    
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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)
    
            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                
                sin_calendar = sin_costfunction(time_dim,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))
    
                name_addedinfo="calday"
                sin_2D = np.tile(sin_calendar, (depth_dim,1))
                cos_2D = np.tile(cos_calendar, (depth_dim,1))
                timeofday_2D = np.tile(timeofday, (depth_dim,1))

                if depan=="linear":
                    name_addedinfo="caldayl"
                    ti = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                if depan=="inverse":
                    name_addedinfo="caldayi"
                    ti = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                if depan=="var":
                    name_addedinfo="caldayv"
                    ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                dt = [0,0,0,0]
    
            
                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time})
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                if vario==True:
                    
                    for single_depth in depth_array:
                        print("Looking at depth: "+str(single_depth ))
                        qs_mean=simulations.sel(depth=single_depth,method="nearest").mean(dim="realizations").squeeze()
                        qs_median=simulations.sel(depth=single_depth,method="nearest").median(dim="realizations").squeeze()
                        print("load the mean and median data")
                        qs_mean=qs_mean.load()
                        qs_median=qs_median.load()
                    
                        sims = np.array([qs_mean,qs_median,simulations_lin.sel(depth=single_depth,method="nearest").squeeze(),simulations_akima.sel(depth=single_depth,method="nearest").squeeze(),simulations_spline.sel(depth=single_depth,method="nearest").squeeze(),simulations_quad.sel(depth=single_depth,method="nearest").squeeze(),simulations_pchip.sel(depth=single_depth,method="nearest").squeeze(),simulations_subdlin.sel(depth=single_depth,method="nearest").squeeze()])
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
                        akima_mean=rmse_var_list[3]
                        spline_mean=rmse_var_list[4]
                        quad_mean=rmse_var_list[5]
                        pchip_mean=rmse_var_list[6]
                        subdlin_mean=rmse_var_list[7]
                        
                        print("variogram done")
                        print(datetime.datetime.now())
                                                 
                                                 
                                           
                    
                        error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                        error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                        error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                        error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                        error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                        error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                    
                        corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                        corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                        corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                        corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                        corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)
    
                        corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                        error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                        std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").mean().values,4)
    
                        df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,member_mean,member_median,lin_mean,akima_mean,spline_mean,quad_mean,pchip_mean,subdlin_mean,std_ratio,single_depth]], columns=df.columns)
                        df = pd.concat([df, df_temp], axis=0)
                
                else:



                
                    error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                    error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                    error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                    error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                    error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                    error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                    corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                    corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                    corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                    corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                    corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)
                    
    
                    corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                    
                    error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                    
                    std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").mean().values,4)
                    df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                    df = pd.concat([df, df_temp], axis=0)
                df.to_csv(output_name, index=False)
                year = 2020
                start_month = 8 
                end_month = 9
                plotting_depth=[-1,-2,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50]
    
                plot_MPS_ensembles_2D(original = data_original.sel(depth=plotting_depth,method="nearest"),
                                      simulation = simulations.sel(depth=plotting_depth,method="nearest"),
                                      year = year,
                                      start_month = start_month,
                                      end_month = end_month,
                                      suptitle = name_addedinfo)
                plotname=plot_folder+name_addedinfo+"run"+str(run)+"N"+str(N)+"pc"+str(percent)+"gap"+str(gap_amount_list[i])
                plt.savefig(plotname+".pdf")
                plt.savefig(plotname+".png")
                plt.show()
    
    
    return simulations,df
    
    
def compare_variograms_nothreads(obs, simlist, decompose_seasonal =False, gap_indices=None, bin_number = None):
    """
    Compare the experimental variogram of observed data to the variograms of a list of simulated data.
    ThreadPoolExecutor() is used to parallelize the experimental variogram computatation. 

    Parameters:
    ----------
    obs : numpy array
        Time series of observed data.

    simlist : list of numpy arrays
        List of time series containing simulated data for comparison.

    decompose_seasonal : bool, optional (default=False)
        If True, decompose the observed data into seasonal trend and residual components before comparing variograms.
        Only the residuals are kept for calculation of the variograms. 

    gap_indices : list of int or None, optional (default=None)
        List of indices corresponding to artifical gaps. If provided, the variograms will only be calculated for these gaps. 

    bin_number : int or None, optional (default=None)
        Number of bins for variogram calculation. If None, an appropriate number of bins will be determined automatically.

    """
    if decompose_seasonal==True:
        if np.count_nonzero(np.isnan(obs))!=0: #decompose function doesn't take nans, so interpolate now and place back the nans later
            gaps = np.isnan(obs)
            obs = obs.interpolate('linear')
        obs_decomposed = sm.tsa.seasonal_decompose(obs, model = 'additive', period = 365)
        trend = obs_decomposed.trend
        obs = obs - trend
        if np.count_nonzero(np.isnan(obs))!=0:
            obs[gaps] = np.nan 
        simlist = [sim - trend for sim in simlist] #to make sure they are detrended in the exact same way
    print("calculating obs variogram")
    bin_center, gamma_obs = exp_variogram(obs, gap_indices=gap_indices, bin_number = bin_number)
    print("calculating sims variogram")
    gamma_sim_list = [exp_variogram(sim, gap_indices=gap_indices, bin_number = bin_number)[1] for sim in simlist]    
    
    return bin_center, gamma_obs, gamma_sim_list
    
def day_of_year_g2s_2D(original,var,obs_in_day,N,percent_list,gap_amount_list,selector_list,test_runs,df,csv_folder,plot_folder,name,depan="linear",vario=True):
    data_original = original[var]
    output_name=csv_folder+name+var+".csv"
    depth_array=[-1.2,-5,-10,-15,-30,-48]
    
    print("metrics saved to: "+output_name)
    if os.path.exists(output_name):
        df=pd.read_csv(output_name)
    
    timeofday = data_original.time.dt.hour.values #C
    runs=np.arange(1,test_runs+1)
    
    depth_dim, time_dim = data_original.shape
    
    depth_linear = np.transpose(np.tile(data_original.depth.data,(time_dim,1)))
    
    depth_inverse = 1/depth_linear
    mask_var=data_original.isnull().sum(dim="time")>(data_original.data.shape[1]/2)
    # where do we have more than 50% nans
    
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
    
    for run in runs:
        for percent in percent_list:
            gap_locations,depth_level_indices,ds24=create_gap_index_nooverlap_2D(da=data_original,gap_percent=percent,gap_length=obs_in_day)
    
            for i in range(len(gap_amount_list)):
                gapped_data=create_gapped_ts_2D(da=data_original,gap_locations=gap_locations,depth_level_index=depth_level_indices,gap_length=gap_amount_list[i],selector=selector_list[i])
                
                sin_calendar = sin_costfunction(time_dim,daily_timesteps = obs_in_day)
                cos_calendar = cos_costfunction(time_dim,daily_timesteps = obs_in_day)
                print("This is run "+str(run)+" with N="+str(N)+" added missing % is "+str(percent)+" and Gap size is "+str(gap_amount_list[i]))
    
                name_addedinfo="calday"
                sin_2D = np.tile(sin_calendar, (depth_dim,1))
                cos_2D = np.tile(cos_calendar, (depth_dim,1))
                timeofday_2D = np.tile(timeofday, (depth_dim,1))

                if depan=="linear":
                    name_addedinfo="caldayl"
                    ti = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_linear,sin_2D, cos_2D],axis = 2)
                if depan=="inverse":
                    name_addedinfo="caldayi"
                    ti = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_inverse,sin_2D, cos_2D],axis = 2)
                if depan=="var":
                    name_addedinfo="caldayv"
                    ti = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                    di = np.stack([gapped_data.data, depth_variance,sin_2D, cos_2D],axis = 2)
                dt = [0,0,0,0]
    
            
                stacked = ensemble_QS(N = N,
                                      ti=ti, 
                                      di=di,
                                      dt=dt, #Zero for continuous variables
                                      k=1.2,
                                      n=50,
                                      j=0.5,
                                      ki=None)
                simulations = xr.DataArray(data =stacked[:,:,:,0],coords = {'realizations':np.arange(1,stacked.shape[0]+1),'depth':data_original.depth.data,'time':gapped_data.time})
                
                
                simulations_lin=gapped_data.interpolate_na(dim="time", method="linear")
                simulations_slin=gapped_data.interpolate_na(dim="time", method="slinear")
                simulations_akima=gapped_data.interpolate_na(dim="time", method="akima")
                simulations_spline=gapped_data.interpolate_na(dim="time", method="spline")
                simulations_quad=gapped_data.interpolate_na(dim="time", method="quadratic")
                simulations_pchip=gapped_data.interpolate_na(dim="time", method="pchip")
                simulations_subdlin=subdaily_linear_interp(gapped_data,times_of_day = obs_in_day)
                
                if vario==True:
                    
                    for single_depth in depth_array:
                        print("Looking at depth: "+str(single_depth ))
                        qs_mean=simulations.sel(depth=single_depth,method="nearest").mean(dim="realizations").squeeze()
                        qs_median=simulations.sel(depth=single_depth,method="nearest").median(dim="realizations").squeeze()
                        print("load the mean and median data")
                        qs_mean=qs_mean.load()
                        qs_median=qs_median.load()
                    
                        sims = np.array([qs_mean,qs_median,simulations_lin.sel(depth=single_depth,method="nearest").squeeze(),simulations_akima.sel(depth=single_depth,method="nearest").squeeze(),simulations_spline.sel(depth=single_depth,method="nearest").squeeze(),simulations_quad.sel(depth=single_depth,method="nearest").squeeze(),simulations_pchip.sel(depth=single_depth,method="nearest").squeeze(),simulations_subdlin.sel(depth=single_depth,method="nearest").squeeze()])
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
                        akima_mean=rmse_var_list[3]
                        spline_mean=rmse_var_list[4]
                        quad_mean=rmse_var_list[5]
                        pchip_mean=rmse_var_list[6]
                        subdlin_mean=rmse_var_list[7]
                        
                        print("variogram done")
                        print(datetime.datetime.now())
                                                 
                                                 
                                           
                    
                        error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                        error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                        error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                        error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                        error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                        error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                    
                        corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                        corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                        corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                        corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                        corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                        corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)
    
                        corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                        error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                        std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").mean().values,4)
    
                        df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,member_mean,member_median,lin_mean,akima_mean,spline_mean,quad_mean,pchip_mean,subdlin_mean,std_ratio,single_depth]], columns=df.columns)
                        df = pd.concat([df, df_temp], axis=0)
                        df.to_csv(output_name, index=False)
                
                else:



                
                    error_lin = np.round(np.sqrt(np.nanmean((simulations_lin.data-data_original.data)**2)),4)
                    error_akima = np.round(np.sqrt(np.nanmean((simulations_akima.data-data_original.data)**2)),4)
                    error_spline = np.round(np.sqrt(np.nanmean((simulations_spline.data-data_original.data)**2)),4)
                    error_quad = np.round(np.sqrt(np.nanmean((simulations_quad.data-data_original.data)**2)),4)
                    error_pchip = np.round(np.sqrt(np.nanmean((simulations_pchip.data-data_original.data)**2)),4)
                    error_subdlin = np.round(np.sqrt(np.nanmean((simulations_subdlin.data-data_original.data)**2)),4)
                
                    corr_lin=np.round(xr.corr(data_original, simulations_lin, dim="time").mean().values,4)
                    corr_akima=np.round(xr.corr(data_original, simulations_akima, dim="time").mean().values,4)
                    corr_spline=np.round(xr.corr(data_original, simulations_spline, dim="time").mean().values,4)
                    corr_quad=np.round(xr.corr(data_original, simulations_quad, dim="time").mean().values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                    corr_pchip=np.round(xr.corr(data_original, simulations_pchip, dim="time").mean().values,4)
                    corr_subdlin=np.round(xr.corr(data_original, simulations_subdlin, dim="time").mean().values,4)
                    
    
                    corr=np.round(xr.corr(data_original, simulations, dim="time").mean(dim="realizations").mean().values,4)
                    
                    error = np.round(np.sqrt(np.nanmean((simulations.data-data_original.data)**2)),4)
                    
                    std_ratio=np.round((data_original/simulations).mean(dim="realizations").mean(dim="time").mean().values,4)
                    df_temp = pd.DataFrame([[name_addedinfo,run, N, percent, gap_amount_list[i], corr,corr_lin,corr_akima,corr_spline,corr_quad,corr_pchip,corr_subdlin,error,error_lin,error_akima,error_spline,error_quad,error_pchip,error_subdlin,std_ratio]], columns=df.columns)
                    df = pd.concat([df, df_temp], axis=0)
                    df.to_csv(output_name, index=False)

    
    
    return simulations,df