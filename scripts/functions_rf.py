#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import cartopy.crs as ccrs
import fsspec
import git 
import json
import sys
import yaml
import requests
import cdsapi
from datetime import timezone
import math 

# check scikit-learn version
import sklearn
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os
from sklearn import preprocessing

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import csv
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator


from functions_verification import *
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#  this just trains on the whole train sample size and gives you one prediciton value
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=2500)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# this selects the respective label data from a pandas data frame. Per default it is chla and 1.4 meter depth
def select_label(df,lookforward_steps,name="chla",depth=1.4):
    naming_list_target=[]
    for i in range(lookforward_steps):
        naming_list_target.append(name+"_"+str(depth)+"t+"+str(i))
    target_selection=df[naming_list_target]
    return target_selection

# this selects the respective input data from a pandas data frame. So everything BUT chla and 1.4 meter depth
# this is not very generalized and depends on the sequence of data that is joined into one data frame
# it also depends on the data source
# maybe there is an easier way to generalize and extract the data
def select_input(df,lookforward_steps):
    concat_list=[]
    for i in range(lookforward_steps):
        if i==0:
            end_name="-29.3t+"
            end_name=end_name+str(i)
            concat_list.append(df.loc[:,:end_name])
        else:
            start_name="lex_u10t+"
            end_name="-29.3t+"
            start_name=start_name+str(i)
            end_name=end_name+str(i)
            concat_list.append(df.loc[:,start_name:end_name])
    training_selection=pd.concat(concat_list, axis=1)
    return training_selection


# In[ ]:


#  this just trains on the whole train sample size
def random_forest_forecast_MW(trainX,trainy, testX):
	# transform list into array
	#train = asarray(train)
	# split into input and output columns
	#trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=2500)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(testX)
	return yhat#[0]


# In[ ]:


#  this gives you the prediction plus the importances and models
def random_forest_forecast_MW_FI(trainX,trainy, testX):
	# transform list into array
	#train = asarray(train)
	# split into input and output columns
	#trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=2500)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(testX)
	importances = list(model.feature_importances_)
	return yhat,importances,model


# this selects the depths we are interested in from the meteolakes dataframe
def choose_depths_meteolakes(df,depth_list):
    new_df=df.iloc[:,0]
    for depth in depth_list:
        location=df.columns.get_loc(int(depth), method='nearest')
        temp_df=df.iloc[:,location]
        new_df=pd.concat([new_df, temp_df],axis=1)
    new_df=new_df.iloc[:,1:]
    return new_df


# In[ ]:


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# In[ ]:

# this creates a feature list from a pandas data frame
def feature_list_lookback_lookforward(names,timesteps,lookforward):
    feature_list=[]
    t_backwards=timesteps
    amount_columns=len(names)
    timestimes=t_backwards*amount_columns
    for i in np.flip(range(t_backwards+1+lookforward)):
        i=(i-lookforward)*-1
        for j in range(amount_columns):
            if i > -1:
                feature_list.append(names[j]+"t+"+str(i))
                
            else:
                feature_list.append(names[j]+"t"+str(i))
    return feature_list


# In[ ]:

# this creates running means from a pandas data frame
def create_running_mean(input_df,timesteps=2,drop_na=True):
    names=list(input_df.columns)
    new_names=[]
    if drop_na==True:
        running_mean=input_df.rolling(timesteps).mean().dropna()
    else:
        running_mean=input_df.rolling(timesteps).mean()
    for i in names:
        i=i+"_rm"+str(timesteps)
        new_names.append(i)
    running_mean.columns = new_names
    return running_mean


# old function not used anymore
def max_norm_df(input_ds,dict_vars,dict_depths="nan",depths=False):
    maxima=[]
    ts=[]
    colnames=[]
    if depths==True:
        for variable in dict_vars:
            for meter in dict_depths:
                colname=variable+"_"+str(meter*-1)
                da=input_ds[variable].sel(depth=meter)
                maximum=da.max()
                max_norm=da/maximum
                maxima.append(maximum.values)
                ts.append(max_norm.values)
                colnames.append(colname)
        ts=np.stack(ts, axis=1)
        maxima=np.stack(maxima)
    else:
        for variable in dict_vars:
            colname=variable
            da=input_ds[variable]
            maximum=da.max()
            max_norm=da/maximum
            maxima.append(maximum.values)
            ts.append(max_norm.values)
            colnames.append(colname)
        ts=np.stack(ts, axis=1)
        maxima=np.stack(maxima)
    return ts,maxima,colnames


# In[ ]:

# this gets the past data from the meteolakes model for the Lexplore location
def get_meteolakes_data_lexplore(start_date=[2022,5,16,21],end_date=[2022,7,16,21],depth=5,variable="temperature",depth_profile=False,today=False):
    
    # based on
    # http://meteolakes.ch/#!/data
    
    easting=540297
    northing=150184
    lake="geneva"
    
    #540'507.10, 150'296.09
    
    if today==False:
        start_date=dt.datetime(start_date[0], start_date[1], start_date[2],start_date[3])
        end_date=dt.datetime(end_date[0], end_date[1], end_date[2],end_date[3])
    else:
        today=datetime.today()- timedelta(days=1)
        start_date=dt.datetime(start_date[0], start_date[1], start_date[2],start_date[3])
        end_date=dt.datetime(today.year, today.month, today.day,21)
    timestamp_start = start_date.replace(tzinfo=timezone.utc).timestamp()
    timestamp_start=int(np.round(timestamp_start,0))
    timestamp_end = end_date.replace(tzinfo=timezone.utc).timestamp()
    timestamp_end=int(np.round(timestamp_end,0))
    url_meteolakes="http://meteolakes.ch/api/coordinates/"+str(easting)+"/"+str(northing)+"/"+lake+"/"+variable+"/"+str(timestamp_start)+"000/"+str(timestamp_end)+"000/"
    if depth_profile==False:
        url_meteolakes=url_meteolakes+str(depth)
    with requests.Session() as s:
        download = s.get(url_meteolakes)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        print("Format of downloaded data is: "+str(np.shape(my_list)))
    cnames=[]
    for i in range(np.shape(my_list)[0]):
        cnames.append(my_list[i][0])
    data=[]
    for i in range(np.shape(my_list)[0]):
        data.append(my_list[i][1:])
    end_date_string=end_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
    start_date_string=start_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
    meteolakes_df=pd.DataFrame(np.transpose(data), index=pd.period_range(start_date_string, end_date_string,freq="3H"))
    meteolakes_df.columns = cnames
    meteolakes_df=meteolakes_df.iloc[:, 1:]
    if np.sum(meteolakes_df.iloc[1,:].values=="NaN")>0:
        meteolakes_df=meteolakes_df.iloc[:, np.sum(meteolakes_df.iloc[1,:].values=="NaN"):]
    meteolakes_df.columns = pd.Float64Index(meteolakes_df.columns)
    print("First Date is: ",meteolakes_df.index.values[0])
    print("Last Date is: ",meteolakes_df.index.values[-1])
    return meteolakes_df


# In[ ]:

# this gets the future data from the meteolakes model for the Lexplore location
def get_datalakes_data_lexplore(date=[2022,5,16,21],future=True):
    easting=540297
    northing=150184
    lake="geneva"
    today=datetime.today()
    future=today + timedelta(days=2)
    sel_date=today    
    if future==False:
        sel_date=dt.datetime(date[0], date[1], date[2],date[3])
    timestamp = sel_date.replace(tzinfo=timezone.utc).timestamp()
    timestamp=int(np.round(timestamp,0))
    
    url_timelines_datalakes="https://api.meteolakes.ch/api/datalakes/timeline/"+lake+"/"+str(timestamp)+"000/"+str(easting)+"/"+str(northing)
    
        # import urllib library
    from urllib.request import urlopen

    # import json
    import json
    # store the URL in url as 
    # parameter for urlopen
    url = url_timelines_datalakes
    # store the response of URL
    response = urlopen(url)

    # storing the JSON response 
    # from url in data
    data_json = json.loads(response.read())

    # print the json response
    #print(data_json)

    temperature=np.transpose(data_json["z"])
    time_list=data_json["x"]
    cnames=np.round(data_json["y"],1)
    start=datetime.utcfromtimestamp(time_list[0]).strftime('%Y-%m-%d %H')
    end=datetime.utcfromtimestamp(time_list[-1]).strftime('%Y-%m-%d %H')
    datalakes_df=pd.DataFrame(temperature, index=pd.period_range(start, end,freq="3H"))
    datalakes_df.columns = cnames
    if np.sum(datalakes_df.iloc[1,:].values==None)>0:
        datalakes_df=datalakes_df.iloc[:, np.sum(datalakes_df.iloc[1,:].values==None):]
    print("First Date is: ",datalakes_df.index.values[0])
    print("Last Date is: ",datalakes_df.index.values[-1])
    return datalakes_df


# In[ ]:


# this turns xarray data into a pandas data frame
# extra name can add additional explanation to the data, e.g. when you have two regions with the same variables, you can name them reg1 and reg2
def xarray_to_df(input_ds,dict_vars,dict_depths="nan",add_name="var",extra_name=False,depths=False,withtime=True):
    ts=[]
    colnames=[]
    if extra_name==False:
        if depths==True:
            for variable in dict_vars:
                for meter in dict_depths:
                    colname=variable+"_"+str(meter*-1)
                    da=input_ds[variable].sel(depth=meter, method='nearest')
                    ts.append(da.values)
                    colnames.append(colname)
            if withtime==True:
                ts.append(input_ds.time.dt.dayofyear)
                colnames.append("dayofyear")
                ts.append(input_ds.time.dt.hour)
                colnames.append("hourofday")
            ts=np.stack(ts, axis=1)
        else:
            for variable in dict_vars:
                colname=variable
                da=input_ds[variable]
                ts.append(da.values)
                colnames.append(colname)
            if withtime==True:
                ts.append(input_ds.time.dt.dayofyear)
                colnames.append("dayofyear")
                ts.append(input_ds.time.dt.hour)
                colnames.append("hourofday")
            ts=np.stack(ts, axis=1)
            
    else:
        if depths==True:
            for variable in dict_vars:
                for meter in dict_depths:
                    colname=add_name+variable+"_"+str(meter*-1)
                    da=input_ds[variable].sel(depth=meter, method='nearest')
                    ts.append(da.values)
                    colnames.append(colname)
            if withtime==True:
                ts.append(input_ds.time.dt.dayofyear)
                colnames.append("dayofyear")
                ts.append(input_ds.time.dt.hour)
                colnames.append("hourofday")
            ts=np.stack(ts, axis=1)
        else:
            for variable in dict_vars:
                colname=add_name+variable
                da=input_ds[variable]
                ts.append(da.values)
                colnames.append(colname)
            if withtime==True:
                ts.append(input_ds.time.dt.dayofyear)
                colnames.append("dayofyear")
                ts.append(input_ds.time.dt.hour)
                colnames.append("hourofday")
            ts=np.stack(ts, axis=1)
    return ts,colnames


# In[ ]:

# old function not used anymore
def ext_max_norm_df(input_ds,dict_vars,df_max,dict_depths="nan",depths=False):
    maxima=[]
    ts=[]
    colnames=[]
    if depths==True:
        for variable in dict_vars:
            for meter in dict_depths:
                index_var=np.where(np.stack(dict_vars) == variable)
                index_depth=np.where(np.stack(dict_depths) == meter)
                #(int(index_var[0])+1)*(int(index_depth[0])+1)
                colname=variable+"_"+str(meter*-1)
                da=input_ds[variable].sel(depth=meter)
                maximum=df_max[colname].values
                max_norm=da/maximum
                #maxima.append(maximum.values)
                ts.append(max_norm.values)
                colnames.append(colname)
        ts=np.stack(ts, axis=1)
    else:
        for variable in dict_vars:
            colname=variable
            da=input_ds[variable]
            maximum=df_max[colname].values
            max_norm=da/maximum
            #maxima.append(maximum.values)
            ts.append(max_norm.values)
            colnames.append(colname)
        ts=np.stack(ts, axis=1)
    return ts,maxima,colnames


# In[ ]:

# this gets the latest hourly era5 data
def get_era5_hourly_lexplore(startyear,area,variables,wd="/Volumes/lexplore_hd/era5/forecast/"):
    os.chdir(wd)
    c = cdsapi.Client()
    today= datetime.today()
    year=today.year
    year_string=str(year)
    year_string_today=year_string
    most_recent_data=today - timedelta(days=7)# the release delay is less than 7 days, but just to be on the safe side
    most_recent_month=most_recent_data.month
    most_recent_year=most_recent_data.year
    most_recent_year_string=str(most_recent_year)
    year_string_today=most_recent_year_string
    if startyear<year:
        year_string=[str(i).zfill(2) for i in range(startyear, most_recent_year+1)]
    most_recent_data_string=most_recent_data.strftime('%Y-%m-%d')
    string_days=[str(i).zfill(2) for i in range(1, most_recent_data.day+1)]
    if most_recent_month==1:
        month_until_complete=12
        #year_string=str(year-1)
    else:
        month_until_complete=most_recent_month-1
    months=[str(i).zfill(2) for i in range(1, month_until_complete+1)]
    months_oldold=[str(i).zfill(2) for i in range(1, 12+1)]
    most_recent_month_string=str(most_recent_month).zfill(2)


    locations=[]
    for i in year_string:
        locations.append(wd+"era5_lexplore_"+i+".nc")
    if most_recent_month==1:
        locations=locations[:-1]
        if path.exists(locations[-1])==True:
            os.remove(locations[-1]) # this is just to make sure we have the complete previous year stored
    locations.append(wd+"era5_lexplore_newest_data.nc")
    
    
    for i in year_string:
            data_location=wd+"era5_lexplore_"+i+".nc"
            newest_data_location=wd+"era5_lexplore_newest_data.nc"
            if year_string_today==i:
                if most_recent_month>1:
                    c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': variables,
                        'year': i,
                        'month': months,
                        'day': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': area,
                        'format': 'netcdf',
                    },
                    data_location)

                c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': variables,
                    'year': str(most_recent_data.year),
                    'month': most_recent_month_string,
                    'day': string_days,
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': area,
                    'format': 'netcdf',
                },
                newest_data_location)

            else:
                if path.exists(data_location)==False:
                    c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': variables,
                        'year': i,
                        'month': months_oldold,
                        'day': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': area,
                        'format': 'netcdf',
                    },
                    data_location)
    merge_list=[] 
    
    locations=[]
    for file in rm_point(os.listdir(wd)):
        da=xr.open_dataset(file,engine="netcdf4")
        if len(da.dims)>3:
            #print(file)
            da=da.mean(("expver"))
            da.to_netcdf(file)
        locations.append(wd+file)
    #print(locations)
    era5=xr.open_mfdataset(locations,engine="netcdf4")
    if len(era5.dims)>3:
        era5=era5.mean(("expver"))
    return era5

# this gets the latest hourly era5_land data
def get_era5_land_hourly_lexplore(startyear,area,variables,wd,endyear=datetime.today().year):
    os.chdir(wd)
    c = cdsapi.Client()
    today= datetime.today()
    year=today.year
    if endyear<year:
        year=endyear
    year_string=str(year)
    year_string_today=year_string
    most_recent_data=today - timedelta(days=7)# the release delay is less than 7 days, but just to be on the safe side
    most_recent_month=most_recent_data.month
    most_recent_year=most_recent_data.year
    most_recent_year_string=str(most_recent_year)
    year_string_today=most_recent_year_string
    if startyear<year:
        year_string=[str(i).zfill(2) for i in range(startyear, most_recent_year+1)]
    most_recent_data_string=most_recent_data.strftime('%Y-%m-%d')
    string_days=[str(i).zfill(2) for i in range(1, most_recent_data.day+1)]
    if most_recent_month==1:
        month_until_complete=12
        #year_string=str(year-1)
    else:
        month_until_complete=most_recent_month-1
    months=[str(i).zfill(2) for i in range(1, month_until_complete+1)]
    months_oldold=[str(i).zfill(2) for i in range(1, 12+1)]
    most_recent_month_string=str(most_recent_month).zfill(2)

    print("the following years are included:")
    print(year_string)
    
    locations=[]
    for v in variables:
        locations.append(wd+"era5_land_lexplore_newest_data_"+v+".nc")
    for i in year_string:
        for m in months:
            for v in variables:
                locations.append(wd+"era5_land_lexplore_"+i+m+"_"+v+".nc")
    #print(locations)
    if most_recent_month==1:
        print("Most recent month is January")
        locations=locations[:-48]
        if path.exists(locations[-48])==True:
            for l in locations[-48:]:
                os.remove(l) # this is just to make sure we have the complete previous year stored
    else:
        print("Most recent month is not January")
    
    
    for i in year_string:
            if year_string_today==i:
                if most_recent_month>1:
                    for m in months:
                        for v in variables:
                            data_location=wd+"era5_land_lexplore_"+i+m+"_"+v+".nc"
                            c.retrieve(
                            'reanalysis-era5-land',
                            {
                                'product_type': 'reanalysis',
                                'variable': v,
                                'year': i,
                                'month': m,
                                'day': [
                                    '01', '02', '03',
                                    '04', '05', '06',
                                    '07', '08', '09',
                                    '10', '11', '12',
                                    '13', '14', '15',
                                    '16', '17', '18',
                                    '19', '20', '21',
                                    '22', '23', '24',
                                    '25', '26', '27',
                                    '28', '29', '30',
                                    '31',
                                ],
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],
                                'area': area,
                                'format': 'netcdf',
                            },
                            data_location)
                for v in variables:
                    newest_data_location=wd+"era5_land_lexplore_newest_data_"+v+".nc"

                    c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'product_type': 'reanalysis',
                        'variable': v,
                        'year': str(most_recent_data.year),
                        'month': most_recent_month_string,
                        'day': string_days,
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': area,
                        'format': 'netcdf',
                    },
                    newest_data_location)

            else:
                for m in months_oldold:
                        for v in variables:
                            data_location=wd+"era5_land_lexplore_"+i+m+"_"+v+".nc"
                            
                            if path.exists(data_location)==False:
                                c.retrieve(
                                'reanalysis-era5-land',
                                {
                                    'product_type': 'reanalysis',
                                    'variable': v,
                                    'year': i,
                                    'month': m,
                                    'day': [
                                        '01', '02', '03',
                                        '04', '05', '06',
                                        '07', '08', '09',
                                        '10', '11', '12',
                                        '13', '14', '15',
                                        '16', '17', '18',
                                        '19', '20', '21',
                                        '22', '23', '24',
                                        '25', '26', '27',
                                        '28', '29', '30',
                                        '31',
                                    ],
                                    'time': [
                                        '00:00', '01:00', '02:00',
                                        '03:00', '04:00', '05:00',
                                        '06:00', '07:00', '08:00',
                                        '09:00', '10:00', '11:00',
                                        '12:00', '13:00', '14:00',
                                        '15:00', '16:00', '17:00',
                                        '18:00', '19:00', '20:00',
                                        '21:00', '22:00', '23:00',
                                    ],
                                    'area': area,
                                    'format': 'netcdf',
                                },
                                data_location)
    locations=[]
    for file in rm_point(os.listdir(wd)):
        da=xr.open_dataset(file,engine="netcdf4")
        if len(da.dims)>3:
            #print(file)
            da=da.mean(("expver"))
            da.to_netcdf(file)
        locations.append(wd+file)
    #print(locations)
    era5_land=xr.open_mfdataset(locations,engine="netcdf4")
    if len(era5_land.dims)>3:
        era5_land=era5_land.mean(("expver"))
    return era5_land



# this uses the data frame input to run several prediction models
def forecast_models_non_iterative(df,list_rolling,lookforward_steps,target_timestep=1,start_N=10,train_test_split=0.8,multi_steps=True,walk_forward=True):
    prediction_list=[]
    measured_list=[]
    modelname_list=[]
    scaler = MinMaxScaler(feature_range=(0, 1))
    cutting_rolling_values=lookforward_steps-1
    label=select_label(df=df,lookforward_steps=lookforward_steps)
    if multi_steps==False:
        label=label.iloc[:,target_timestep-1]
    inputs=select_input(df=df,lookforward_steps=lookforward_steps)
    inputs_incl_rolling=pd.concat([inputs,list_rolling[0].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    for i in np.arange(1,len(list_rolling)):
        inputs_incl_rolling=pd.concat([inputs_incl_rolling,list_rolling[i].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    feature_list=list(inputs_incl_rolling.columns)
    
    label_values=label.values
    input_values=inputs_incl_rolling.values
    input_values_NN=scaler.fit_transform(input_values)
    label_values_NN=scaler.fit_transform(label_values.reshape(-1, 1))
    n_records = len(input_values)
    
    print("Total Sample Size: "+str(n_records))
    
    print("Total Feature Size "+str(len(feature_list)))

    split_point=int(len(input_values)*train_test_split)
    
    print("Training Sample Size: "+str(split_point))
    
    if multi_steps==False:
        print("Forecast Goal: T+"+str(target_timestep))
        
    if multi_steps==True:
        print("Forecast Goal: T+1 to T+"+str(lookforward_steps))
    
    trainX=input_values[:split_point,:]
    trainy=label_values[:split_point]
    testX=input_values[split_point:,:]
    persistence=label_values[split_point-1:-1]
    testy=label_values[split_point:]
    
    trainX_NN=input_values_NN[:split_point,:]
    trainy_NN=label_values_NN[:split_point]
    testX_NN=input_values_NN[split_point:,:]
    testy_NN=label_values_NN[split_point:]
    
    if walk_forward==False:
        forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
        feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        print("Top5 Important Features")
        print(feature_importances[:5])
        prediction_list.append(forecast)
        
        modelname_list.append("RF")
        
        regr = linear_model.LinearRegression(n_jobs=100)
        regr.fit(trainX, trainy)
        preds_reg = regr.predict(testX)
        prediction_list.append(preds_reg)
        
        modelname_list.append("LREG")
       
        if multi_steps==False:
            from sklearn import svm
            regr_svr = svm.SVR()
            regr_svr.fit(trainX, trainy)
            preds_svr=regr_svr.predict(testX)#
            prediction_list.append(preds_svr)
            modelname_list.append("SVM")
        
        
        
        regr_MLP = MLPRegressor(hidden_layer_sizes=(64),
                       max_iter = 3000,activation = 'relu',
                       solver = 'adam').fit(trainX, trainy)
        preds_MLP=regr_MLP.predict(testX)
        
        prediction_list.append(preds_MLP)
        modelname_list.append("MLP")
        
        #ess = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
       # ess_val = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
      #  dense_64 = Sequential()
       # dense_64.add(Dense(64, activation='relu', input_dim=trainX.shape[1]))
      #  if len(trainy.shape)==1:
      #      dense_64.add(Dense(1, activation='relu'))
      #  else:
     #       dense_64.add(Dense(trainy.shape[1], activation='relu'))
     #   dense_64.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
      #  dense_64.fit(trainX_NN, trainy_NN, batch_size=32, epochs=3000, verbose=0,validation_split=0.1, callbacks=[ess_val])
    
      #  preds_NN=dense_64.predict(testX_NN)
        
      #  prediction_list.append(preds_NN)
       #  modelname_list.append("NN")
    #
        

        
        measured_list=testy
        
        if multi_steps==True:
            print("prediction_list contains RF, LREG, and MLP predictions")
            
        else:
            print("prediction_list contains RF, LREG, SVM and MLP predictions")
    
    else:
        prediction_list=[]
        measured_list=[]
        error_rf_list=[]
        error_per_list=[]
        error_LREG_list=[]
        error_SVM_list=[]
        error_MLP_list=[]
        error_NN_list=[]
        modelname_list=[]
        modelname_list.append("RF")
        modelname_list.append("LREG")
        if multi_steps==False:
            modelname_list.append("SVM")
        modelname_list.append("MLP")
        print("Walk Forward Validation Initialized")

        for i in range(start_N, n_records):
            print("------------------")
            trainX, testX = input_values[0:i], input_values[i:i+1]
            trainy,testy= label_values[0:i], label_values[i:i+1]
            persistence=label_values[i-1:i]
            print('train=%d, test=%d' % (len(trainX), len(testX)))
            forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
            
            regr = linear_model.LinearRegression(n_jobs=100)
            regr.fit(trainX, trainy)
            preds_reg = regr.predict(testX)
            
            if multi_steps==False:
                from sklearn import svm
                regr_svr = svm.SVR()
                regr_svr.fit(trainX, trainy)
                preds_svr=regr_svr.predict(testX)
            
            regr_MLP = MLPRegressor(hidden_layer_sizes=(64),
                       max_iter = 3000,activation = 'relu',
                       solver = 'adam').fit(trainX, trainy)
            preds_MLP=regr_MLP.predict(testX)
            
            #ess = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        #    ess_val = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
         #   dense_64 = Sequential()
          #  dense_64.add(Dense(64, activation='relu', input_dim=trainX.shape[1]))
          #  if len(trainy.shape)==1:
           #     dense_64.add(Dense(1, activation='relu'))
           # else:
           #     dense_64.add(Dense(trainy.shape[1], activation='relu'))
          #  dense_64.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
           # dense_64.fit(trainX_NN, trainy_NN, batch_size=32, epochs=3000, verbose=0,validation_split=0.1, callbacks=[ess_val])
           # preds_NN=dense_64.predict(testX_NN)
            
            print("expected value:"+str(testy))
            print("predicted value RF:"+str(forecast))
            print("predicted value LREG:"+str(preds_reg))
            if multi_steps==False:
                print("predicted value SVM:"+str(preds_svr))
            print("predicted value MLP:"+str(preds_MLP))
           # print("predicted value NN:"+str(preds_NN))
            errors_RF = abs(forecast - testy)
            errors_LREG = abs(preds_reg - testy)
            if multi_steps==False:
                errors_SVM = abs(preds_svr - testy)
            errors_MLP = abs(preds_MLP - testy)
          #  errors_NN = abs(scaler.inverse_transform(preds_NN) - scaler.inverse_transform(testy_NN))
            errors_pers = abs(persistence - testy)
            prediction_list.append(forecast)
            
            measured_list.append(testy)
            feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the mean absolute error (mae)
            print('Mean Absolute Error:', round(np.mean(errors_RF), 2),"RF")
            error_rf_list.append(round(np.mean(errors_RF), 2))
            print('Mean Absolute Error:', round(np.mean(errors_LREG), 2),"LREG")
            error_LREG_list.append(round(np.mean(errors_LREG), 2))
            if multi_steps==False:
                print('Mean Absolute Error:', round(np.mean(errors_SVM), 2),"SVM")
                error_SVM_list.append(round(np.mean(errors_SVM), 2))
            print('Mean Absolute Error:', round(np.mean(errors_MLP), 2),"MLP")
            error_MLP_list.append(round(np.mean(errors_MLP), 2))
          #  print('Mean Absolute Error:', round(np.mean(errors_NN), 2),"NN")
          #  error_NN_list.append(round(np.mean(errors_NN), 2))
            print('Mean Absolute Error:', round(np.mean(errors_pers), 2),"PER")
            error_per_list.append(round(np.mean(errors_pers), 2))
            print("Top5 Important Features:")
            print(feature_importances[:5])
            
    return forecast,importances,model,feature_importances,measured_list,prediction_list,modelname_list

# this uses the data frame input to run a random forest prediction model

def rf_non_iterative(df,list_rolling,lookforward_steps,target_timestep=1,start_N=10,train_test_split=0.8,multi_steps=True,walk_forward=True):
    cutting_rolling_values=lookforward_steps-1
    label=select_label(df=df,lookforward_steps=lookforward_steps)
    if multi_steps==False:
        label=label.iloc[:,target_timestep-1]
    inputs=select_input(df=df,lookforward_steps=lookforward_steps)
    inputs_incl_rolling=pd.concat([inputs,list_rolling[0].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    for i in np.arange(1,len(list_rolling)):
        inputs_incl_rolling=pd.concat([inputs_incl_rolling,list_rolling[i].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    feature_list=list(inputs_incl_rolling.columns)
    
    label_values=label.values
    input_values=inputs_incl_rolling.values
    n_records = len(input_values)
    
    print("Total Sample Size: "+str(n_records))
    
    print("Total Feature Size "+str(len(feature_list)))

    split_point=int(len(input_values)*train_test_split)
    
    print("Training Sample Size: "+str(split_point))
    
    if multi_steps==False:
        print("Forecast Goal: T+"+str(target_timestep))
        
    if multi_steps==True:
        print("Forecast Goal: T+1 to T+"+str(lookforward_steps))
    
    trainX=input_values[:split_point,:]
    trainy=label_values[:split_point]
    testX=input_values[split_point:,:]
    persistence=label_values[split_point-1:-1]
    testy=label_values[split_point:]
    
    if walk_forward==False:
        forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
        feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        print("Top5 Important Features")
        print(feature_importances[:5])
        prediction_list=forecast
        measured_list=testy
        
    else:
        prediction_list=[]
        measured_list=[]
        error_rf_list=[]
        error_per_list=[]
        print("Walk Forward Validation Initialized")

        for i in range(start_N, n_records):
            print("------------------")
            trainX, testX = input_values[0:i], input_values[i:i+1]
            trainy,testy= label_values[0:i], label_values[i:i+1]
            persistence=label_values[i-1:i]
            print('train=%d, test=%d' % (len(trainX), len(testX)))
            forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
            
            print("expected value:"+str(testy))
            print("predicted value:"+str(forecast))
            errors = abs(forecast - testy)
            errors_pers = abs(persistence - testy)
            prediction_list.append(forecast)
            measured_list.append(testy)
            feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the mean absolute error (mae)
            print('Mean Absolute Error:', round(np.mean(errors), 2),"RF")
            error_rf_list.append(round(np.mean(errors), 2))
            print('Mean Absolute Error:', round(np.mean(errors_pers), 2),"PER")
            error_per_list.append(round(np.mean(errors_pers), 2))
            print("Top5 Important Features:")
            print(feature_importances[:5])
            
    return forecast,importances,model,feature_importances,measured_list,prediction_list

# this plots the data made by the prediction routines above
def forecast_plotting(prediction_list,measured_list,modelname_list):
    xaxis=np.arange(0,np.shape(measured_list)[0],1)
    if len(np.shape(measured_list))>1:
        for i in range(np.shape(measured_list)[1]):
            fig = plt.figure(figsize=(8, 4), dpi= 200)
            plt.plot(xaxis,measured_list[:,i],label="measured")
            plt.title("t+"+str(i+1))
            print("t+"+str(i+1))
            for m in range(len(modelname_list)):               
                plt.plot(xaxis,prediction_list[m][:,i],label=modelname_list[m])
                errors = abs(prediction_list[m][:,i] - measured_list[:,i])
                print('Mean Absolute Error:', round(np.mean(errors), 2),modelname_list[m])
                correlation=np.corrcoef(prediction_list[m][:,i], measured_list[:,i])
                print('Correlation:',np.round(correlation[0,1],2),modelname_list[m])
            plt.legend()
    if len(np.shape(measured_list))==1:
        fig = plt.figure(figsize=(8, 4), dpi= 200)
        plt.plot(xaxis,measured_list[:],label="measured")
        for m in range(len(modelname_list)):               
            plt.plot(xaxis,prediction_list[m][:],label=modelname_list[m])
            errors = abs(prediction_list[m][:] - measured_list[:])
            print('Mean Absolute Error:', round(np.mean(errors), 2),modelname_list[m])
            correlation=np.corrcoef(prediction_list[m][:], measured_list[:])
            print('Correlation:',np.round(correlation[0,1],2),modelname_list[m])
        plt.legend()
    return plt.show()

# this uses the data frame input to run a random forest prediction model, but it uses the predicted t+1 to predict t+2 etc etc
# see the power point presentation for more
def rf_iterative(df,list_rolling,lookforward_steps,train_test_split=0.8):
    cutting_rolling_values=lookforward_steps-1
    list_of_forecasts=[]
    label_list=[]
    testy_list=[]
    additive_list=[]
    
    for i in range(lookforward_steps):
        label_list.append(df["chla_1.4t+"+str(i)])
        
    for i in np.arange(1,lookforward_steps):
        
        additive_list.append(df.loc[:,"lex_u10t+"+str(i):"-29.3t+"+str(i)])
        
    
    
    #----------------- 1st forecast
    
    
    label=label_list[0]
    inputs=df.loc[:,:"-29.3t+0"]
    
    inputs=pd.concat([inputs,list_rolling[0].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    for i in np.arange(1,len(list_rolling)):
        inputs=pd.concat([inputs,list_rolling[i].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    
    
    inputs_values=inputs.values
    label_values=label.values
    feature_list=list(inputs.columns)
    
    split_point=int(len(inputs_values)*train_test_split)
    
    
    trainX=inputs_values[:split_point,:]
    trainy=label_values[:split_point]
    testX=inputs_values[split_point:,:]
    persistence=label_values[split_point-1:-1]
    testy=label_values[split_point:]
    
    
    forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
    list_of_forecasts.append(forecast)
    testy_list.append(testy)
    print("forecast done")
    print("first forecasted value: "+str(forecast[0]))
    
    #----------------- n forecast
    
    for i in np.arange(1,lookforward_steps):
        
        forecast=pd.DataFrame(forecast,columns=[label.name])
        
        if i==1:
            first=pd.concat([inputs,additive_list[i-1]], axis=1)
            recorded_input_for_pred=pd.concat([first.iloc[split_point:,:].reset_index().iloc[:,1:],forecast],axis=1)
        else:
            recorded_input_for_pred=pd.concat([recorded_input_for_pred,additive_list[i-1].iloc[split_point:,:].reset_index().iloc[:,1:],forecast],axis=1)
            
        inputs=pd.concat([inputs,additive_list[i-1],label], axis=1)
        
        label=label_list[i]   
        
        inputs_values=inputs.values
        label_values=label.values
        feature_list=list(inputs.columns)

        trainX=inputs_values[:split_point,:]
        trainy=label_values[:split_point]
        testX=inputs_values[split_point:,:]
        persistence=label_values[split_point-1:-1]
        testy=label_values[split_point:]

        forecast,importances,model=random_forest_forecast_MW_FI(trainX=trainX,trainy=trainy, testX=testX)
    
        input_for_pred_values=recorded_input_for_pred.values
        forecast = model.predict(input_for_pred_values)
        list_of_forecasts.append(forecast)
        testy_list.append(testy)
        print("forecast done")
        print("first forecasted value: "+str(forecast[0]))
    
    return list_of_forecasts,testy_list

# function to resample everything in a factor
def resampling(input_data,time_res="4H"):
    for i in range(len(input_data)):
        if type(input_data[i])==xarray_type:
            input_data[i]=input_data[i].resample(time='4H').mean()
        if type(input_data[i])==pandas_df_type:
            input_data[i]=input_data[i].resample("4H").mean()
    return input_data

# this uses the data frame input to run a LSTM prediction model, whithoutn running means though
# this one uses the data frame WITHOUT lookback values and you specify the lookback window in this function
def lstm_with_lookback(df,lookforward_steps,lookback_steps,target_timestep=1,train_test_split=0.8,nmodels=3,neurons=64,multi_steps=False):
    ess = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    scaler = MinMaxScaler(feature_range=(0, 1))
    inputs=select_input_lstm(df=df,lookforward_steps=lookforward_steps)
    labels=select_label_lstm(df=df,lookforward_steps=lookforward_steps)
    if multi_steps==False:
        labels=labels.iloc[:,target_timestep-1]
        labels=labels.values.reshape(-1, 1)
    input_values=scaler.fit_transform(inputs)
    label_values=scaler.fit_transform(labels)


    n_records = len(input_values)
    split_point=int(len(input_values)*train_test_split)
    trainX=input_values[:split_point,:]
    trainy=label_values[:split_point]
    testX=input_values[split_point:,:]
    persistence=label_values[split_point-1:-1]
    testy=label_values[split_point:]
    train_generator = TimeseriesGenerator(trainX, trainy, length=lookback_steps, batch_size=20)     
    test_generator = TimeseriesGenerator(testX, testy, length=lookback_steps, batch_size=1)


    prediction_list=[]
    model_name_list=[]
    model_list=[]


    for i in np.arange(0,nmodels):
        testy=label_values[split_point:]
        model_name="LSTM"+str(i)
        from keras.models import Sequential
        from keras.layers import LSTM, Dense

        model = Sequential()
        model.add(
            LSTM(neurons,
                activation='relu',
                input_shape=(lookback_steps,trainX.shape[1]))
        )
        model.add(Dense(trainy.shape[1]))
        model.compile(optimizer='adam', loss='mse')

        num_epochs = 2000
        model.summary()
        history=model.fit_generator(train_generator, epochs=num_epochs, verbose=1, callbacks=[ess])
        prediction = model.predict_generator(test_generator)
        prediction=scaler.inverse_transform(prediction)
        testy=scaler.inverse_transform(testy)
        testy=testy[lookback_steps:,:]
        errors = abs(prediction - testy)
        print('Mean Absolute Error:', round(np.mean(errors), 2),model_name)
        prediction_list.append(prediction)
        model_name_list.append(model_name)
        model_list.append(model)
        acc = history.history['loss']
        #val = history_model_lookback.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, '-', label='Training loss')
        #plt.plot(epochs, val, ':', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        plt.plot()
    return model_list,prediction_list,model_name_list,testy

# similar to "select_input" but for the LSTM prediction
def select_input_lstm(df,lookforward_steps):
    concat_list=[]
    for i in range(lookforward_steps):
        if i==0:
            end_name="do_30t+"
            end_name=end_name+str(i)
            concat_list.append(df.loc[:,:end_name])
        else:
            start_name="lex_u10t+"
            end_name="-29.3t+"
            start_name=start_name+str(i)
            end_name=end_name+str(i)
            concat_list.append(df.loc[:,start_name:end_name])
    training_selection=pd.concat(concat_list, axis=1)
    return training_selection

# similar to "select_label" but for the LSTM prediction
def select_label_lstm(df,lookforward_steps,name="chla",depth=1.4):
    naming_list_target=[]
    for i in np.arange(1,lookforward_steps):
        naming_list_target.append(name+"_"+str(depth)+"t+"+str(i))
    target_selection=df[naming_list_target]
    return target_selection

# this uses the data frame input to run a LSTM prediction model, whithoutn running means though
# this one uses the data frame WITH lookback values 
def lstm_without_lookback(df,lookforward_steps,target_timestep=1,train_test_split=0.8,nmodels=3,neurons=64,multi_steps=False):

    ess_val = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    scaler = MinMaxScaler(feature_range=(0, 1))
    inputs=select_input(df=df,lookforward_steps=lookforward_steps)
    labels=select_label(df=df,lookforward_steps=lookforward_steps)
    feature_list=list(inputs.columns)
    if multi_steps==False:
        labels=labels.iloc[:,target_timestep-1]
        labels=labels.values.reshape(-1, 1)
    input_values=scaler.fit_transform(inputs)
    label_values=scaler.fit_transform(labels)
    split_point=int(len(input_values)*train_test_split)

    n_records = len(input_values)

    trainX=input_values[:split_point,:]
    trainy=label_values[:split_point]
    testX=input_values[split_point:,:]
    persistence=label_values[split_point-1:-1]
    testy=label_values[split_point:]

    prediction_list=[]
    model_name_list=[]
    model_list=[]
    for i in np.arange(0,nmodels):
        testy=label_values[split_point:]
        from keras.models import Sequential
        from keras.layers import LSTM, Dense

        model_name="LSTM"+str(i)

        model_nolookback_1 = Sequential()
        model_nolookback_1.add(
            LSTM(neurons,
                activation='relu',
                input_shape=(1,trainX.shape[1]))
        )
        model_nolookback_1.add(Dense(trainy.shape[1]))
        model_nolookback_1.compile(optimizer='adam', loss='mse')

        num_epochs = 2000
        model_nolookback_1.summary()


        history_model_nolookback_1=model_nolookback_1.fit(trainX.reshape(trainX.shape[0],1,trainX.shape[1]), trainy, batch_size=32, epochs=3000, verbose=1,validation_split=0.1, callbacks=[ess_val])
        prediction_nolookback_1=model_nolookback_1.predict(testX.reshape(testX.shape[0],1,testX.shape[1]))

        prediction_nolookback_1=scaler.inverse_transform(prediction_nolookback_1)
        testy=scaler.inverse_transform(testy)
        errors = abs(prediction_nolookback_1 - testy)
        acc = history_model_nolookback_1.history['loss']
        val = history_model_nolookback_1.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, '-', label='Training loss')
        plt.plot(epochs, val, ':', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        plt.plot()
        prediction_list.append(prediction_nolookback_1)
        model_name_list.append(model_name)
        model_list.append(model_nolookback_1)
        print('Mean Absolute Error:', round(np.mean(errors), 2),model_name)
    return model_list,prediction_list,model_name_list,testy

# this plots the outcome of the RF iterative forecast
def forecast_plotting_iterative(prediction_list,measured_list,allinone=False):
    measured_list=np.asarray(measured_list)
    prediction_list=np.asarray(prediction_list)
    xaxis=np.arange(0,np.shape(measured_list)[1],1)
    for i in range(np.shape(measured_list)[0]):
        fig = plt.figure(figsize=(8, 4), dpi= 200)
        plt.plot(xaxis,measured_list[i,:],label="measured")
        plt.title("t+"+str(i+1))
        print("t+"+str(i+1))            
        plt.plot(xaxis,prediction_list[i,:],label="predicted")
        errors = abs(prediction_list[i,:] - measured_list[i,:])
        print('Mean Absolute Error:', round(np.mean(errors), 2),"RF")
        correlation=np.corrcoef(prediction_list[i,:], measured_list[i,:])
        print('Correlation:',np.round(correlation[0,1],2),"RF")
        plt.legend()
    if allinone==True:
        colors=["yo","go","co","bo","mo","ro"]
        measured_list=np.asarray(measured_list)
        prediction_list=np.asarray(prediction_list)
        xaxis=np.arange(0,np.shape(measured_list)[1],1)
        fig = plt.figure(figsize=(8, 4), dpi= 200)
        for start in range(np.shape(measured_list)[1]-np.shape(measured_list)[0]):
            for i in range(len(prediction_list[:,0])):
                plt.plot(xaxis[i+start],prediction_list[i,start],colors[i],alpha=0.2)
        plt.plot(xaxis[:],measured_list[0,:],'ko',label="measured")
        plt.legend()
        
    return plt.show()

# this plots the outcome of the LSTM forecasts
def forecast_plotting_lstm(prediction_list,measured_list,modelname_list):
    xaxis=np.arange(0,np.shape(measured_list)[0],1)
    for i in range(np.shape(measured_list)[1]):
        fig = plt.figure(figsize=(8, 4), dpi= 200)
        plt.plot(xaxis,measured_list[:,i],"ko",label="measured")
        plt.title("t+"+str(i+1))
        print("t+"+str(i+1))
        for m in range(len(modelname_list)):               
            plt.plot(xaxis,prediction_list[m][:,i],label=modelname_list[m])
            errors = abs(prediction_list[m][:,i] - measured_list[:,i])
            print('Mean Absolute Error:', round(np.mean(errors), 2),modelname_list[m])
            correlation=np.corrcoef(prediction_list[m][:,i], measured_list[:,i])
            print('Correlation:',np.round(correlation[0,1],2),modelname_list[m])
        plt.legend()
    
    return plt.show()

# lets you make the prediction from a trained model.

def rf_prediction_only(df,list_rolling,lookforward_steps,model,target_timestep=1,multi_steps=False):
    prediction_list=[]
    measured_list=[]
    modelname_list=[]
    cutting_rolling_values=lookforward_steps-1
    label=select_label(df=df,lookforward_steps=lookforward_steps)
    if multi_steps==False:
        label=label.iloc[:,target_timestep-1]
    inputs=select_input(df=df,lookforward_steps=lookforward_steps)
    inputs_incl_rolling=pd.concat([inputs,list_rolling[0].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    for i in np.arange(1,len(list_rolling)):
        inputs_incl_rolling=pd.concat([inputs_incl_rolling,list_rolling[i].iloc[:-cutting_rolling_values,:].reset_index().iloc[:,1:]],axis=1)
    feature_list=list(inputs_incl_rolling.columns)
    
    label_values=label.values
    input_values=inputs_incl_rolling.values
    n_records = len(input_values)
    
    print("Total Sample Size: "+str(n_records))
    
    print("Total Feature Size "+str(len(feature_list)))
    
    X=input_values
    y=label_values
    prediction=model.predict(X)
    return prediction,y

def get_historical_meteolakes_lexplore(startyear,endyear,output_folder,xcoord="540996",ycoord="150144",lake="geneva",var="temperature",csv_save=True,nc_save=True):
    
    #try:
    list_of_meteolakes_chunks=[]
    today= datetime.today()
    for year in range(startyear,endyear+1):
        print("Downloading year "+str(year))
        range_months=range(1,12+1)
        if year == today.year:
            range_months=range(1,today.month-1)
        for month in range_months:

            if month==12:
                start_date=[year,month,1,3]
                end_date=[year+1,1,1,0]
            else:
                start_date=[year,month,1,3]
                end_date=[year,month+1,1,0]

            print("Downloading month "+str(month))

            start_date=dt.datetime(start_date[0], start_date[1], start_date[2],start_date[3])
            end_date=dt.datetime(end_date[0], end_date[1], end_date[2],end_date[3])
            timestamp_start = start_date.replace(tzinfo=timezone.utc).timestamp()
            timestamp_start=int(np.round(timestamp_start,0))
            timestamp_end = end_date.replace(tzinfo=timezone.utc).timestamp()
            timestamp_end=int(np.round(timestamp_end,0))
            url_meteolakes="http://meteolakes.ch/api/coordinates/"+xcoord+"/"+ycoord+"/"+lake+"/"+var+"/"+str(timestamp_start)+"000/"+str(timestamp_end)+"000/"
            with requests.Session() as s:
                download = s.get(url_meteolakes)
            print(download)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            #print(np.shape(my_list))
            if np.shape(my_list)!=(1,2):
                cnames=[]
                for i in range(np.shape(my_list)[0]):
                    cnames.append(my_list[i][0])
                data=[]
                for i in range(np.shape(my_list)[0]):
                    data.append(my_list[i][1:])
                end_date_string=end_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
                start_date_string=start_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
                #print(np.shape(data))
                meteolakes_df=pd.DataFrame(np.transpose(data), index=pd.period_range(start_date_string, end_date_string,freq="3H"))
                meteolakes_df.columns = cnames
                meteolakes_df=meteolakes_df.iloc[:, 1:]
                if np.sum(meteolakes_df.iloc[1,:].values=="NaN")>0:
                    meteolakes_df=meteolakes_df.iloc[:, np.sum(meteolakes_df.iloc[1,:].values=="NaN"):]
                meteolakes_df.columns = pd.Float64Index(meteolakes_df.columns)
                list_of_meteolakes_chunks.append(meteolakes_df)
                #print(meteolakes_df)
            else:
                print("Data is missing for this month")
                end_date_string=end_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
                #print(end_date_string)
                start_date_string=start_date.strftime('%Y-%m-%d %H')#('%d-%m-%y %H')
                #print(start_date_string)
                index_time=pd.period_range(start_date_string, end_date_string,freq="3H")
                #print(np.shape(index_time))
                index_length=len(index_time)
                nans=np.repeat(math.nan,index_length*60)
                data=np.reshape(nans,(index_length,60))
                data=np.transpose(data)
                #print(np.shape(data))
                meteolakes_df=pd.DataFrame(np.transpose(data), index=index_time)
                meteolakes_df.columns = cnames
                meteolakes_df=meteolakes_df.iloc[:, 1:]
                if np.sum(meteolakes_df.iloc[1,:].values=="NaN")>0:
                    meteolakes_df=meteolakes_df.iloc[:, np.sum(meteolakes_df.iloc[1,:].values=="NaN"):]
                meteolakes_df.columns = pd.Float64Index(meteolakes_df.columns)
                list_of_meteolakes_chunks.append(meteolakes_df)
                #print(meteolakes_df)
                    
    #except: 
    #    print(str(year)+" does not have all values available")

    name=xcoord+"_"+ycoord+"_"+lake+"_"+var+"_"+str(timestamp_start)+"000_"+str(timestamp_end)+"000"
    meteolakes_yearly_df=pd.concat(list_of_meteolakes_chunks,axis=0)
    
    meteolakes_yearly_df.index=meteolakes_yearly_df.index.astype('datetime64[ns]') 
    meteolakes_yearly_df = meteolakes_yearly_df.apply(pd.to_numeric,errors='ignore')
    if csv_save==True:
        meteolakes_yearly_df.to_csv(output_folder+name+".csv")
        print("Data written to "+output_folder+name+".csv")

    if var=="temperature":
        metadata=dict(description="Water Temperature",units="degC",)

    da = xr.DataArray(data=meteolakes_yearly_df.values,dims=["time","depth"],coords=dict(depth=meteolakes_yearly_df.columns,time=meteolakes_yearly_df.index),attrs=metadata,)
    if nc_save==True:
        nc_location=output_folder+name+".nc"
        da.to_netcdf(nc_location)
        print("Data written to "+nc_location)
    else:
        nc_location="no .nc data written out"
    return da,nc_location
