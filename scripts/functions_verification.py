#!/usr/bin/env python
# coding: utf-8


# these are the functions used for the data verification script for lexplore data

import numpy as np
import xarray as xr
import netCDF4
import os
import datetime
import matplotlib.pyplot as plt 
from matplotlib import cm
import shutil
from datetime import datetime, timedelta
import glob
import datetime as dt
from os import path
import fsspec
import json
import sys
import yaml




#def clean_thetis(thetis_folder,new_folder):
#    if path.exists(new_folder):
#        shutil.rmtree(new_folder)
#    shutil.copytree(thetis_folder, new_folder)
#    if path.exists(new_folder+"L2_THETIS_GRID_20210425_20210505.nc"):
#        for file_name in os.listdir(new_folder):
#            source = new_folder + file_name
#            source_split=source.split("L2_THETIS_GRID_")
#            new="L2_"+source_split[1]
#            new=new_folder+new
#            os.rename(source, new)
#            #os.remove(source)
#    if path.exists(new_folder+"L2_20220609_20220619.nc"):
#        os.remove(new_folder+"L2_20220609_20220619.nc")
#
#   if path.exists(new_folder+"L2_20191202_20191212.nc"):
#        os.remove(new_folder+"L2_20191202_20191212.nc")
#    return print("Corrupt Thetis files removed")

#def clean_thetis(thetis_folder):
#    if path.exists(os.path.join(thetis_folder,"L2_THETIS_GRID_20220609_20220619.nc")):
#        os.remove(thetis_folder+"L2_THETIS_GRID_20220609_20220619.nc")
#
#    if path.exists(thetis_folder+"L2_THETIS_GRID_20191202_20191212.nc"):
#        os.remove(thetis_folder+"L2_THETIS_GRID_20191202_20191212.nc")
#    return print("Corrupt Thetis files removed")


#def clean_pp(thetis_folder,new_folder):
#    if path.exists(new_folder):
#        shutil.rmtree(new_folder)
#    shutil.copytree(thetis_folder, new_folder)
#    for file_name in os.listdir(new_folder):
#        source = new_folder + file_name
#        source_split=source.split("L2_")
#        new="L2"+source_split[1]
#        new=new_folder+new
#        os.rename(source, new)
#            #os.remove(source)
#   return print("PP Mooring files renamed")


def get_night(infile):
    evening=infile.where(infile['time.hour'] >21, drop=True)
    morning=infile.where(infile['time.hour'] <7, drop=True)
    output=xr.merge([morning,evening])
    return output


def obs_per_hour(infile):
    obs_amount=[]
    for i in range(24):
        amountsobs=len(infile.where(infile['time.hour'] ==i, drop=True).time.values)
        obs_amount.append(amountsobs)
        print("Amount of obs for time "+str(i)+": "+str(amountsobs))
    return print("These are the times sampled for "+infile.source)


def time_filter_files(start, end, files):
    if files[0].startswith("L1"):
        f = []
        for file in files:
            splitnc=file.split(".nc")
            split_length=len(splitnc[0].split("_"))
            remove = False
            dt = datetime.strptime(splitnc[0].split("_")[split_length-1], '%Y%m%d')
            if start and dt < start:
                remove = True
            if end and dt > end:
                remove = True
            if not remove:
                f.append(file)
    else:
        f = []
        for file in files:
            split_length=len(file.split("_"))
            remove = False
            dt = datetime.strptime(file.split("_")[split_length-2], '%Y%m%d')
            if start and dt < start:
                remove = True
            if end and dt > end:
                remove = True
            if not remove:
                f.append(file)
    return f



def pres2depth(pressure):
    depth=pressure/(9.80655*997.0474/1000) # depth in meter, pressure in kPa
    return depth # for freshwater





def read_in_multiple_files(startyear,endyear,folder,pr=True):
    nyears=endyear-startyear+1
    folder_content=os.listdir(folder)

    start = datetime(startyear, 1, 1)
    end = datetime(endyear, 12, 31)

    if start:
         s_string = start.strftime("%Y%m%d")
    else:
        s_string = "Start"
    if end:
        e_string = end.strftime("%Y%m%d")
    else:
        e_string = "End"
    list_of_starts=[]
    list_of_ends=[]
    for i in np.arange(startyear,endyear+1):
        start = datetime(i, 1, 1)
        end = datetime(i, 12, 31)
        list_of_starts.append(start)
        list_of_ends.append(end)
    
    if path.exists(os.path.join(folder,"L2_THETIS_GRID_20220609_20220619.nc")):
        folder_content.remove("L2_THETIS_GRID_20220609_20220619.nc")
    if path.exists(os.path.join(folder,"L2_THETIS_GRID_20191202_20191212.nc")):
        folder_content.remove("L2_THETIS_GRID_20191202_20191212.nc")
    files_unsorted = rm_point(folder_content)
    for i in range(len(list_of_starts)):
        year = str(list_of_starts[i].year)
        name = "files_sorted_"
        locals()[name + year] =  sorted(time_filter_files(list_of_starts[i], list_of_ends[i], files_unsorted))
    for i in range(len(list_of_starts)):
        year = str(list_of_starts[i].year)
        name = "files_sorted_"
        for i in range(len(locals()[name + year])):
            locals()[name + year][i]=(os.path.join(folder, locals()[name + year][i]))
    list_of_instrument=[]
    for i in range(len(list_of_starts)):
        year = str(list_of_starts[i].year)
        name = "instrument_"
        name2 = "files_sorted_"
        if locals()[name2 + year]:
            if pr==True:
                print("Reading in Year: "+year)
            locals()[name + year]=xr.open_mfdataset(locals()[name2 + year],combine="nested",concat_dim="time")
            data=xr.open_mfdataset(locals()[name2 + year],combine="nested",concat_dim="time")
            list_of_instrument.append(data)
            if pr==True:
                print("amount of time steps in year available: "+str(len(data.time.values)))
    for i in range(len(list_of_instrument)):
        list_of_instrument[i]=list_of_instrument[i].sel(time=~list_of_instrument[i].get_index("time").duplicated())
    instrument_all=xr.merge(list_of_instrument)
    if pr==True:
        print("amount of total time steps available: "+str(len(instrument_all.time.values)))
    return instrument_all




def nofeb(file):
    nofeb=file.sel(time=~((file.time.dt.month == 2) & (file.time.dt.day == 29)))
    return nofeb





def rm_point(files):
    f=[]
    for i in range(len(files)):
        if not files[i].startswith("."):
            f.append(files[i])
    return f





def get_time(file):
    time=file.time.values
    day_in_year=[]
    hour=[]
    date_index=[]
    for i in range(len(time)):
        t = np.datetime64(time[i])
        t1 = dt.datetime.utcfromtimestamp(t.tolist() / 1e9)
        day_in_year.append(t1.timetuple().tm_yday)
        hour.append(t1.hour)
        if t1.day>9:
            if t1.month>9:
                date_index.append(int(str(t1.year)+str(t1.month)+str(t1.day)))
            else:
                date_index.append(int(str(t1.year)+str(0)+str(t1.month)+str(t1.day)))
        else:
            if t1.month>9:
                date_index.append(int(str(t1.year)+str(t1.month)+str(0)+str(t1.day)))
            else:
                date_index.append(int(str(t1.year)+str(0)+str(t1.month)+str(0)+str(t1.day)))
            
        #day_in_year=np.array(day_in_year)
        #hour=np.array(hour)
    return day_in_year,date_index,hour

def get_time_fish(file):
    time=file.ping_time.values
    day_in_year=[]
    hour=[]
    date_index=[]
    for i in range(len(time)):
        t = np.datetime64(time[i])
        t1 = dt.datetime.utcfromtimestamp(t.tolist() / 1e9)
        day_in_year.append(t1.timetuple().tm_yday)
        hour.append(t1.hour)
        if t1.day>9:
            if t1.month>9:
                date_index.append(int(str(t1.year)+str(t1.month)+str(t1.day)))
            else:
                date_index.append(int(str(t1.year)+str(0)+str(t1.month)+str(t1.day)))
        else:
            if t1.month>9:
                date_index.append(int(str(t1.year)+str(t1.month)+str(0)+str(t1.day)))
            else:
                date_index.append(int(str(t1.year)+str(0)+str(t1.month)+str(0)+str(t1.day)))
            
        #day_in_year=np.array(day_in_year)
        #hour=np.array(hour)
    return day_in_year,date_index,hour


def rm_time_duplicates(file):
    file.sel(time=~file.get_index("time").duplicated())
    return file

def verification(dataset1,dataset2):
    #get time values
    dataset1_time=dataset1.time.values
    dataset2_time=dataset2.time.values
    #rounding to the closest hour
    dataset1_hour=np.array(dataset1_time, dtype='datetime64[h]')
    dataset2_hour=np.array(dataset2_time, dtype='datetime64[h]')
    # precise overlap over the next hour
    time_intersect=np.intersect1d(dataset1_hour,dataset2_hour)
    # select common overlap time steps
    dataset1_dataset2_intersect_nearest=dataset1.sel(time=time_intersect, method='nearest')
    dataset2_dataset1_intersect_nearest=dataset2.sel(time=time_intersect, method='nearest')
    return dataset1_dataset2_intersect_nearest,dataset2_dataset1_intersect_nearest


def ver_plotting_pp(dataset1,dataset2,var1,output_folder,cc="depth"):
    z=dataset1.depth.values
    z_scaled=(z-z.min())/z.ptp()
    colors=plt.cm.viridis(z_scaled,len(z))
    day_in_year,date_index,hour=get_time(dataset1)
    day_scaled=(np.array(day_in_year)-min(day_in_year))/np.ptp(day_in_year)
    colors_day=plt.cm.viridis(day_scaled)
    date_scaled=(np.array(date_index)-min(date_index))/np.ptp(date_index)
    colors_date=plt.cm.viridis(date_scaled)
    #
    if cc=="date":
        amount_dates=len(date_index)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_dates):
            plt.scatter(dataset1[var1].sel(depth=-5.0, method='nearest')[i],dataset2["5.0m"][i],color=colors_date[i],alpha=0.5, label=date_index[i])
            plt.scatter(dataset1[var1].sel(depth=-10.0, method='nearest')[i],dataset2["10.0m"][i],color=colors_date[i],alpha=0.5, label=date_index[i])
            plt.scatter(dataset1[var1].sel(depth=-30.0, method='nearest')[i],dataset2["30.0m"][i],color=colors_date[i],alpha=0.5, label=date_index[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name+" 5, 10, 30m"
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
    else:
        fig = plt.figure(figsize=(13, 7))
        plt.scatter(dataset1[var1].sel(depth=-5.0, method='nearest'),dataset2["5.0m"],label="5m",alpha=0.5)
        plt.scatter(dataset1[var1].sel(depth=-10.0, method='nearest'),dataset2["10.0m"],label="10m",alpha=0.5)
        plt.scatter(dataset1[var1].sel(depth=-30.0, method='nearest'),dataset2["30.0m"],label="30m",alpha=0.5)

        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name+" 5, 10, 30m"
        color_code=cc
        plt.title(title)
        
    plt.xlabel(dataset1.source+" "+dataset1[var1].units)
    plt.ylabel(dataset2.source+" "+dataset2.units)
    if cc=="depth":
        plt.legend()
    if cc=="date":
        cbar.ax.set_yticklabels([min(date_index),max(date_index)]) 
    xpoints=ypoints=plt.xlim()
    plt.plot(xpoints,ypoints,color="k",lw=3,scalex=False,scaley=False,alpha=0.2)
    fig.savefig(os.path.join(output_folder,title+" "+color_code+".pdf"))
    plt.close('all')
    return print("Figure saved: "+title+" "+color_code+".pdf")


def ver_plotting(dataset1,dataset2,var1,var2,output_folder,cc="depth"):
    z=dataset1.depth.values
    z_scaled=(z-z.min())/z.ptp()
    colors=plt.cm.viridis(z_scaled,len(z))
    day_in_year,date_index,hour=get_time(dataset1)
    day_scaled=(np.array(day_in_year)-min(day_in_year))/np.ptp(day_in_year)
    colors_day=plt.cm.viridis(day_scaled)
    date_scaled=(np.array(date_index)-min(date_index))/np.ptp(date_index)
    colors_date=plt.cm.viridis(date_scaled)
    hour_scaled=(np.array(hour)-min(hour))/np.ptp(hour)
    colors_hour=plt.cm.viridis(hour_scaled)
    
    if cc=="date":
        amount_dates=len(date_index)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_dates):
            plt.scatter(dataset1[var1][:,i],dataset2[var2][:,i],color=colors_date[i],alpha=0.5, label=date_index[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
    if cc=="depth":
        amount_depth=len(z)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_depth):
            plt.scatter(dataset1[var1][i,:],dataset2[var2][i,:],color=colors[i],alpha=0.5, label=z[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
    if cc=="hour":
        amount_depth=len(hour)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_depth):
            plt.scatter(dataset1[var1][:,i],dataset2[var2][:,i],color=colors_hour[i],alpha=0.5, label=hour[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
        
    plt.xlabel(dataset1.source+" "+dataset1[var1].units)
    plt.ylabel(dataset2.source+" "+dataset2[var2].units)
    #plt.legend()
    if cc=="date":
        cbar.ax.set_yticklabels([min(date_index),max(date_index)]) 
    if cc=="depth":
        cbar.ax.set_yticklabels([min(z),max(z)])
    if cc=="hour":
        cbar.ax.set_yticklabels([min(hour),max(hour)]) 
    xpoints=ypoints=plt.xlim()
    plt.plot(xpoints,ypoints,color="k",lw=3,scalex=False,scaley=False,alpha=0.2)
    fig.savefig(os.path.join(output_folder,title+" "+color_code+".pdf"))
    plt.close('all')
    return print("Figure saved: "+title+" "+color_code+".pdf")


def ver_plotting_tchain(dataset1,dataset2,var1,output_folder,cc="depth"):
    z=dataset1.depth.values
    z_scaled=(z-z.min())/z.ptp()
    colors=plt.cm.viridis(z_scaled,len(z))
    day_in_year,date_index,hour=get_time(dataset1)
    day_scaled=(np.array(day_in_year)-min(day_in_year))/np.ptp(day_in_year)
    colors_day=plt.cm.viridis(day_scaled)
    date_scaled=(np.array(date_index)-min(date_index))/np.ptp(date_index)
    colors_date=plt.cm.viridis(date_scaled)
    
    if cc=="date":
        amount_dates=len(date_index)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_dates):
            plt.scatter(dataset1[var1][:,i],dataset2[:,i],color=colors_date[i],alpha=0.5, label=date_index[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
    else:
        amount_depth=len(z)
        fig = plt.figure(figsize=(13, 7))
        for i in range(amount_depth):
            plt.scatter(dataset1[var1][i,:],dataset2[i,:],color=colors[i],alpha=0.5, label=z[i])
        title=dataset1.source+" vs "+dataset2.source+" "+ dataset1[var1].long_name
        plt.title(title)
        color_code=cc
        cbar=plt.colorbar(ticks=[0, 1], label=color_code)
        
    plt.xlabel(dataset1.source+" "+dataset1[var1].units)
    plt.ylabel(dataset2.source+" "+dataset2.units)
    #plt.legend()
    if cc=="date":
        cbar.ax.set_yticklabels([min(date_index),max(date_index)]) 
    else:
        cbar.ax.set_yticklabels([min(z),max(z)])
    xpoints=ypoints=plt.xlim()
    plt.plot(xpoints,ypoints,color="k",lw=3,scalex=False,scaley=False,alpha=0.2)
    fig.savefig(os.path.join(output_folder,title+" "+color_code+".pdf"))
    plt.close('all')
    return print("Figure saved: "+title+" "+color_code+".pdf")

def plot_clim(dataset,var,output_folder,cmap="inferno",res="monthly"):
    if res=="daily":
        dataset_clim=dataset.groupby('time.dayofyear').mean(dim='time')
    if res=="monthly":
        dataset_clim=dataset.groupby('time.month').mean(dim='time')
    fig = plt.figure(figsize=(13, 7))
    if var=="tchain":
        tplot=dataset_clim.transpose().plot.pcolormesh(levels = 17, cmap=plt.cm.get_cmap(cmap, 21))
        label=dataset.units
        title=dataset.source+" "+res+" "+ dataset.long_name+" Climatology"
    else:
        tplot=dataset_clim[var].transpose().plot.pcolormesh(levels = 17, cmap=plt.cm.get_cmap(cmap, 21))
        label=dataset[var].units
        title=dataset.source+" "+res+" "+ dataset[var].long_name+" Climatology"
    tplot.colorbar.set_label(label, size=10) 
    plt.title(title)
    fig.savefig(os.path.join(output_folder,title+".pdf"))
    plt.close('all')
    return print("Figure saved: "+title+".pdf") 

def plot_samples(dataset,var,output_folder,cmap="viridis",res="monthly"):
    if res=="daily":
        dataset_amount=dataset.resample(time='1D').count()
    if res=="monthly":
        dataset_amount=dataset.resample(time='1M').count()
    if res=="yearly":
        dataset_amount=dataset.resample(time='1Y').count()
    fig = plt.figure(figsize=(13, 7))
    if var=="tchain":
        tplot=dataset_amount.plot.pcolormesh(levels = 17, cmap=plt.cm.get_cmap(cmap, 21))
        label="amount of "+res+" observations"
        title=dataset.source+" "+res+" "+ dataset.long_name+" sample size"
    else:
        tplot=dataset_amount[var].transpose().plot.pcolormesh(levels = 17, cmap=plt.cm.get_cmap(cmap, 21))
        label="amount of "+res+" observations"
        title=dataset.source+" "+res+" "+ dataset[var].long_name+" sample size"
    tplot.colorbar.set_label(label, size=10) 
    plt.title(title)
    fig.savefig(os.path.join(output_folder,title+".pdf"))
    plt.close('all')
    return print("Figure saved: "+title+".pdf")

def clean_idronaut(dataset,dates2exclude):
    excl=[]
    for i in range(len(dates2exclude["start"])):
        timeslice=slice(dates2exclude["start"][i], dates2exclude["end"][i])
        excl=dataset.sel(time=timeslice).time.values
        dataset=dataset.drop_sel(time=excl)
    return dataset

def log(str, indent=0, start=False):
    if start:
        out = "\n" + str + "\n"
        with open("log.txt", "w") as file:
            file.write(out + "\n")
    else:
        out = datetime.now().strftime("%H:%M:%S.%f") + (" " * 3 * (indent + 1)) + str
        with open("log.txt", "a") as file:
            file.write(out + "\n")
    print(out)
    
def clean_pp(dataset,var="do"):
    levels=["5.0m","10.0m","30.0m"]
    dataset=dataset[levels]
    if var=="do":
        dataset["5.0m"]=dataset["5.0m"].where(dataset["5.0m"].values<100)
        dataset["10.0m"]=dataset["10.0m"].where(dataset["10.0m"].values<100)
        dataset["30.0m"]=dataset["30.0m"].where(dataset["30.0m"].values<100)
        dataset=dataset[levels]
    if var=="par":
        dataset["5.0m"]=dataset["5.0m"].where(dataset["5.0m"].values<1250)
        dataset["10.0m"]=dataset["10.0m"].where(dataset["10.0m"].values<600)
        dataset["30.0m"]=dataset["30.0m"].where(dataset["30.0m"].values<100)
        
    return dataset