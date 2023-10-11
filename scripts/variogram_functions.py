
from concurrent.futures import ThreadPoolExecutor
import gstools as gs
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


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
        future_obs = executor.submit(exp_variogram, obs, gap_indices=gap_indices, bin_number = bin_number)
        
        # Calculate the experimental variograms for simlist in parallel
        futures_sim = [executor.submit(exp_variogram, sim, gap_indices = gap_indices, bin_number = bin_number) for sim in simlist]
        
        # Retrieve results when ready
        bin_center, gamma_obs = future_obs.result()
        gamma_sim_list = []
        for future in futures_sim:
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

    
    