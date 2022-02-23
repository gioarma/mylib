from nptdms import TdmsFile
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import re
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from scipy.optimize import root
import warnings
import scipy
from natsort import natsorted
import glob
from scipy.constants import physical_constants
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from mylib.plotting import cmap as cm
import hvplot.pandas
import holoviews as hv
import panel as pn
from datetime import datetime
import hvplot.xarray
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import rasterize
import matplotlib.colors as mcolors

###### Physical constants   #######################
k_B = physical_constants['Boltzmann constant'][0]
q = physical_constants['elementary charge'][0]
h = physical_constants['Planck constant'][0]
m_e = physical_constants['electron mass'][0]
###################################################



####### DATA IMPORT/EXPORT  #######################################################################

def read_transients (path, amplifier_gain=1, dropna=False, set_timetrack = True, drop=None):

    '''
    path: string with file path of TDMS file
    amplifier_gain: gain set on the current amplifier (to extract current from voltage values)
    current_scale: Order of magnitude of the currents to scale the data (e.g. 1e-9 sets values to nA). Set 1 to leave as measured
    time_scale: Order of magnitude of the times to scale the data (e.g. 1e-6 sets values to µA). Set 1 to leave as measured
    dropna: whether to drop all rows where there is at least one NaN. Useful for faulty dataset where there are some columns with more data than others
    set_timetrack: whether to get the timetrack from the files. In some corrupted data it is better to avoid it
    drop: list of 2 integers indicating the initial and final columns to be dropped from the dataframe as soon as it is imported. Usually used to remove the first or last columns which may contain corrupted data. If drop=[0,5] drop the first 2 columns
    '''

    if '.pkl' in path: return pd.read_pickle(path,'bz2')   # If transient is passed as compressed pickle file as returned by save_transients
    
    tdms_file = TdmsFile.read(path)
    df = tdms_file['Measured Data'].as_dataframe()    # convert the 'Measured Data' group to a dataframe with all transients

    if dropna: df=df.dropna()

    if drop is not None: df = df.drop(axis=1, columns = df.columns[drop[0]:drop[1]])

    # Temperature values
    df.columns = [float(temp.replace('wf_','')) for temp in df.columns]
    df.columns.name = 'Temperature (K)'

    # Current values
    df=df/amplifier_gain                # Convert to current

    # Time values
    if set_timetrack:
        df.index = tdms_file['Measured Data'].channels()[0].time_track() # Take index from the first channel, it should be the same for all channels (i.e.) temperatures
    #df.index = df.index.values.round(3)
    df.index.name = 'Time (s)'
    # Set t=0 on trigger
    trigger = tdms_file['Measured Data'].channels()[0].properties['wf_trigger_offset']    # LV program sves in wf_trigger_offset the time of the LED triger
    df.index-=trigger
    return df



def save_transients (tr, path):
    '''
    Exports the transient as a pickle file with bz2 compression. This typically allows to decrease of at least 1 order of magnitude the file dimension
    The bz2 compression seems to have the best compromise between file size, read time and write time.\n\n
    tr: Dataframe to be exported\n\n
    path: string containing path/file_name where to save the file.
    '''  
    if '.pkl' not in path:  path += '.pkl'         # Add format, if not already written by the user
    return tr.to_pickle(path, compression='bz2')


def read_temp_ramp (path):
    '''
    Returns dataframe with temperature ramp of the specified file
    '''
    tdms_file = TdmsFile.read(path)                   # Read file
    temp = tdms_file['Temperature'].as_dataframe()
    temp.columns = ['Temperature (K)']
    temp['Time (min)'] = tdms_file['TimeRead'].as_dataframe()['Untitled'].values    # Create new column with read time values
    t0 = np.datetime64('1904-01-01 00:00:00.000000')                                # The datetime values are referred to this date by LabView
    temp['Time (min)'] = (temp['Time (min)'] - t0).dt.total_seconds()/60            # Subtract the time from the reference date to get meaningful data for total seconds, then convert to minutes
    temp = temp.set_index('Time (min)')                                             # Set time as index

    return temp



def save_arrhenius (arr, sample_name, scan_number, gates_number , trap_params=None,
                    integral=None, measurement_date = None,
                    bias=None, xray_dose=None, excitation_wavelength = None,
                    light_source_current=None, T_min=None, T_max=None, heating_rate=None,
                    path = '', append_current_time=False, comment=''):
    
    '''
    Saves the arrhenius plots in a csv file for further analysis.
    
    arr: tuple returned by arrhenius_fit. Alternatively, DataFrame/list of DataFrames. Each DataFrame should contain the arrhenius plots in form of 1000/T as index, ln(T2/en) as columns\n\n
    sample_name: string containing the sample name (use just the code, not any pre-code like 'MAPbBr' or else)\n\n
    scan_number: int indicating the number of temperature scan for the sample, if it's the first T scan, then scan_number=1 and so on. \n\n
    gates_number: number of gates used for obtaining the spectrum (2 or 4).\n\n
    trap_params: second element returned by arrhenius_fit. This has to be passed if arr is a DataFrame or list of DataFrames containing just the arrhenius plots with no info on trap parameters.\n\n
    integral: bool saying if the picts method was standard (integral=False) or integral (integral=True)\n\n
    measurement date: string containing the measurement date expressed as 'dd/mm/yyyy'. \n\n
    bias: bias (in V) applied to the sample during measurement. \n\n
    xray_dose: total X-ray dose (in Gy) delivered to the sample. \n\n 
    excitation_wavelength: wavelength (in nm) of the light source. \n\n 
    light_source_current: current (in mA) flowing through the LED/LASER. \n\n
    T_min: minimum temperature (in K) reached during the scan.
    T_max: maximum temperature (in K) reached during the scan.
    heating_rate: heating rate (in K/min) of the scan.
    path: path where to save the data. If not specified, the csv is saved in the working directory. \n\n
    append_current_time: whether you want to append to the end of the file name the date of when the file was saved. This allows to avoid overwriting when you save twice a file with the same parameters.\n
    comment: string with any kind of additional info to be appended to the file name
    '''
    # If user diddn't put / at the end of path, we add it
    if path != '':
        if path[-1] != '/': path = path+'/'
        
    # If arr is a list, we check that arr and trap_params contain the same number of elements
    if isinstance(arr, list): 
        if trap_params is not None: 
            if len(arr) != len(trap_params): raise ValueError("arr list and trap_params lists do not have the same size.")
        
        if isinstance(arr[0], tuple):    # if a list of tuples is passed. This is the case of when several outputs from arrhenius_fit or map_fit are passed
            params, arrs = [], []        # lists that will be filled with the tuple elements and then assigned to trap_params and arrs. 
            for a in arr:               
                params.append(a[3])
                arrs.append(a[0])
            trap_params = params 
            arr = arrs
                
    # If arr is a single df, we put it into a 1-element list 
    if isinstance(arr, pd.DataFrame):
        arr = [arr]
    if isinstance(trap_params, pd.DataFrame):
        trap_params = [trap_params]
    # If arr is a tuple returned by arrhenius_fit, we extract from it the parameters we need to export
    if isinstance(arr, tuple):
        trap_params = [arr[3]]
        arr = [arr[0]]
    # Get current time expressed from year to seconds
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Create full path + file name
    filename = path + 'arrhenius_' + sample_name + '_Scan'+ str(scan_number) + '_' + str(gates_number) + 'gates'
    if integral: filename+='_integral'
    else: filename+='_standard'
    if append_current_time: filename += '_'+current_time
    if comment != '': filename += '_'+comment
    filename+='.csv'
    
    arr_stacked = [a.stack().reset_index(0).rename(columns={0:'ln(T²/en)'}) for a in arr]     # Create a df with default index and 3 columns: 1000/T, Trap name and ln(ln(T²/en))
    arr_all = pd.concat(arr_stacked, axis=0)  # concatenate all arr vertically
    trap_params_stacked = [t.T for t in trap_params]
    trap_params_all = pd.concat(trap_params_stacked)   # concatenate all arr vertically
    
    df = arr_all.join(trap_params_all)
    # Change formatting of some column names so that it's easier to work on the dataframe later
    df = df.rename(columns = {
        'Eₐ (eV)': 'Ea (eV)',
        'δEₐ (eV)': 'dEa (eV)',
        'σ (cm²)': 'sigma (cm2)',
        'δσ (cm²)': 'dsigma (cm2)'
    })
    # Add other info to the dataframe:
    df['Date saved'] = current_time
    df['Date measured'] = measurement_date
    df['Sample name'] = sample_name
    df['Number of gates'] = gates_number
    df['Integral'] = integral
    df['Scan number'] = scan_number
    df['X-ray dose (Gy)'] = xray_dose
    df['Bias (V)'] = bias
    df['Excitation wavelength (nm)'] = excitation_wavelength
    df['Light source current (mA)'] = light_source_current
    df['T min (K)'] = T_min
    df['T max (K)'] = T_max
    df['Heating rate (K/min)'] = heating_rate
    # Save the csv file
    df.to_csv(filename)
    

def import_arrhenius(path, sample_info_path=None):
    
    '''
    Imports the arrhenius plot data, getting all the sample information from the file name and from the excel file with further info on the samples\n\n
    
    path: string containing the path to the folder where the csv files f the arrhenius plots are stored.\n\n
    sample_info_path: string containing the path to the excel file that contains all the additional info on the samples
    
    '''    
    if path[-1]!='/': path += '/'
    file_list=natsorted(glob.glob(path+'arr*'))  # Find all files in the folder
    
    ## Concatenate all data in a single DataFrame
    arr = pd.concat([pd.read_csv(f) for f in file_list ])
    # Fix some columns that are not read correctly 
    arr['Sample name'] = arr['Sample name'].astype(str)      # convert sample name to string, since it can be seen as an integer by read_csv and create problems when joining dataframes afterwards
    arr['Number of gates'] = arr['Number of gates'].astype(str)   # More convenient for plots and panels
    # We convert this into a multiindex dataframe so that we can join it with sample_info after. We have to do this because arr and sample_info do not have the same amount of rows for each sample (sample_info has only 1 row for each sample)
    arr['count'] = arr.groupby('Sample name').cumcount()        # Create a count column, which is the inner level of the multiindex. It's simply a count of each repetition of Sample name
    arr.set_index(['Sample name','count'], inplace=True)        # The outer level of the multiindex is Sample name
    # Adjust arr columns 
    arr['Date measured'] = pd.to_datetime(arr['Date measured'], dayfirst=True)   # Convert measurement date in datetime format
    
    if sample_info_path is not None:
        ## Get other info from the excel file
        sample_info = pd.read_excel(sample_info_path, engine='openpyxl', index_col='Sample name')
        sample_info.index = sample_info.index.astype(str)       # On excel the sample name may be considered as a int instead of a string if there are no letters in the name, so we convert it to string
        
        df = arr.join(sample_info).reset_index().drop('count', axis=1)   # joining the two dfs, then resetting index and dropping count (I needed them just to perferm the join smoothly)
        df = df.rename(columns={'Unnamed: 0': 'Trap'})  # Trap column was unnamed from read csv, so we give it a name here
        df['Sample age (days)'] = (df['Date measured']-df['Date growth']).astype('timedelta64[h]')/24   # Age of the sample when measured. We convert it to days otherwise in the plots we get the time in nanoseconds, which is a mess
    else: 
        df = arr
    
    # Create new columns
    df['T range (K)'] = df['T max (K)']-df['T min (K)']   # Age of the sample when measured
    df['Bias sign'] = np.where(df['Bias (V)']>0, '+','-')
    return df



def save_map (S, path, sample_name, gates_number, integral, scan_number, comment=''):
    '''
    Saves the 3D Arrhenius map DataArray as a netCDF file.\n\n
    S: xarray.DataArray as returned by picts_map. \n
    path: string containing path where to save the file. \n
    sample_name: string containing the sample name \n
    gates_number: number of gates used for obtaining the map (2 or 4).\n
    integral: bool saying if the picts method was standard (integral=False) or integral (integral=True). \n
    scan_number: int indicating the number of temperature scans for the sample, if it's the first T scan, then scan_number=1 and so on. \n
    comment: string with any kind of additional info to be appended to the file name
    '''
    
    if path[-1] != '/': path+='/'
    path = path + '/' + 'map_' + sample_name + '_Scan' + str(scan_number) + '_' + str(gates_number) + 'gates'
    if not isinstance(integral,bool): raise TypeError("integral keyword must be of boolean type.")
    if integral : path += '_integral'
    else: path += '_standard'
    if comment != '': path += '_'+comment
    path += '.nc'         # Add format, if not already written by the user
    return S.to_netcdf(path) 


def import_map (path):
    return xr.open_dataarray(path)



###############################################################################################################




###############################################################################################################
###### DATA ANALYSIS####################################################################
###############################################################################################################


#### UTILITIES #############################################################################

def drop_T (tr, T_list):
    '''
    Delete desired temperature columns from transient dataframe\n\n
    
    tr: dataframe containing current transients with time as index and temperature as columns.\n\n
    T_list: list of int/lists containing temperature values to be removed. If int, the specific temperature is dropped. If list it has to contain 2 values and all temperatures between these two values are dropped.
    '''
    for T in T_list:
        if isinstance(T, (int,float)): tr = tr.drop(T, axis=1)
        if isinstance(T, list): tr = tr.drop(tr.loc[:,T[0]:T[1]].columns, axis=1)
    return tr


def normalize_transients (tr, i_0_range, i_inf_range, info = False):
    '''
    tr: dataframe with transients at different temperatures
    i_0_range: list of 2 values specifying the time range where to calculate the current when LED is on (typically close to the time where LED is turned off)
    i_inf_range: list of 2 values specifying the time range where to calculate the current when LED is off (typically close to the time where LED is turned on)
    info: if True, returns (normalized transients, i_0, i_inf) otherwise just the normalized transients

    Calculates i_0 and i_inf as the mean in the ranges specified by the user and then normalized from 0 to 1 all transients
    '''

    i_0 = tr.loc[i_0_range[0]:i_0_range[1]].mean()
    i_inf = tr.loc[i_inf_range[0]:i_inf_range[1]].mean()
    tr_norm = (tr-i_inf)/(i_0-i_inf)

    if info:
        return tr_norm, i_0, i_inf
    else:
        return tr_norm


def create_t1_values (t1_min, t1_shift, n_windows = None, tr = None, multiplier = None, method = 'linear'):
    '''
    Creates the set of t1 values that represent the first gates of all rate windows.\n\n
    t1_min: minimum value of t1\n
    t1_shift: for the linear method it represents the "delta t" between one gate and the next\n
    n_windows: nuber of rate windows, i.e. number of t1 values of the returned array. If it's not passed, the max number of windows for the transient is automatically generated. In this case, tr needs to be passed. \n
    tr: Dataframe containing the transients. Has to be passed if n_windows is None, i.e. if the max number of rate windows is automatically generated.\n
    multiplier: should be beta in the case of 2 gates and gamma for 4 gates. Needs to be passed only if n_windows is None.\n
    method: accepted options are 'linear' (linearly increase the t1 values) \n\n
    Returns:
    numpy array with all t1 values.
    '''
    # Determine if the t1 values will automatically generated (auto) or not (user)
    if n_windows is None: mode = 'auto'
    else: mode = 'user'
    
    if mode == 'user':    
        if method == 'linear':
            t1 =  np.array([t1_min+t1_shift*i for i in range(n_windows)])        
        elif method == 'exp':
            t1 = np.array([t1_min*np.exp(t1_shift*i) for i in range(n_windows)])    
        else:
            raise NotImplementedError("This method is not implemented for the moment!")
            
    if mode == 'auto':
        t_max = tr.index.max()     # maximum legal time value
        t1 = np.array([t1_min])
        n_windows=0
        while multiplier*t1[-1] < t_max:    # We fill t1 increasing the number of windows until we hit the highest possible value of t2
            n_windows += 1
            if method == 'linear':
                t1 = np.append(t1, t1_min+t1_shift*n_windows)
            if method == 'exp':
                t1 = np.append(t1, t1_min*np.exp(t1_shift*n_windows))
        t1 = t1[:-1]        # Remove the last item, which will be higher than t_max. This is due to the condition I put in the while loop.
        
    return t1


def round_rate_window_values (df, en, round_value):
    '''
    df: input dataframe where columns are supposed to be en values\n
    en: rate window values\n
    round_value: decimal position en windows should be rounded to\n\n
    Returns:
    Dataframe with column values that are rate windows which have been rounded to the desired value.
    '''
    if round_value is None:
        df.columns = en
    else:
        if round_value>0: df.columns = en.round(round_value)
        elif (round_value==0): df.columns = en.round(0).astype(int)
        else :
            warnings.warn("Negative value of round_en! setting default values of rate windows", stacklevel=2)
            df.columns = en

    return df

def en_2gates_high_injection (en, t1, t2):
    '''
    The roots of this function gives the value of en for a given t1 and t2.
    This is a trascendental equation with 2 solutions. One solution is 0, the other is the real value of en.
    For reference see Balland et al. 1984 part II and Supporting info of Pecunia et al. 2021.
    '''
    return np.exp(en*(t2-t1)) - ( (1-en*t2)/(1-en*t1))


def calculate_en (t1, t2, injection):
    '''
    Returns the rate window values starting from the gate values. In the case of high injection, it numerically solves the related equation
    t1: numpy array coontaining values of the 1st gate \n
    t2: numpy array containing values for the second gate \n
    injection: can be either "high" or "low", corresponding to high or low injection from the light source. The expression for finding en is different in the 2 cases. \n
    \n\n

    Returns: a numpy array with the rate window values
    '''
    if injection == 'high':
        en = np.array([])
        for t1, t2 in zip(t1,t2):
            en_guess = 1/(t2-t1)*(t2/t1)    # As a guess we use this, which seems to work well (totally empiric, 1/(t2-t1) alone sometimes does not work). The problem is we need to choose a starting point that is closer to our searched value than to 0, otherwise the function will return the 0 value as result.
            # We use the root function from scipy.optimize to numerically solve
            en = np.append(en, root(en_2gates_high_injection,
                                    x0=en_guess, args=(t1, t2)).x)
    elif injection == 'low':
        en = np.log(t2/t1)/(t2-t1)
    else:
        raise ValueError('Unknown kind of injection. It can be either "high" or "low".')

    return en


def remove_outliers (df, window, threshold):
    '''
    Substitutes outlier points in a DataFrame with NaN values according to the following algorithm:
    1) On each column of the DataFrame, it calculates the rolling median of the dataframe with the desired window, 
    2) Takes the difference between the original and the rolling median,
    3) Turns into NaNs the points where the difference is bigger than the desired threshold.
    
    df: DataFrame to be smoothed.\n
    window: rolling median window.\n
    threshold: threshold difference between original data and averaged data above which a point is considered outlier.
    '''
    median = df.rolling(window,center=True).median().fillna(method='bfill').fillna(method='ffill')
    difference = abs(df - median)           # calculate diff between original df and the median
    outlier_idx = difference < threshold    # find which points have to be kept, i.e. only those for which the differece is smaller than threshold
    filtered_df = df[outlier_idx]
    return filtered_df
    
###################################################################################
######## SPECTRUM CALCULATION  ####################################################
###################################################################################



def picts_2gates (tr, beta, t_avg, t1_min=None, t1_method='linear', 
                  t1_shift=None, n_windows=None, t1=None, integrate = False, 
                  round_en = None, injection = 'high'):
    '''
    tr: dataframe with transients at different temperatures\n
    t1_min: minimum value of t1. Necessary if t1 is not provided\n
    t1_shift: for the linear method it represents the "delta t" between one gate and the next
    n_windows: nuber of rate windows, i.e. number of t1 values of the returned array \n
    t1_method: method used to create t1 values. Accepted options are 'linear' (linearly increase the t1 values)
    t1: numpy array of values of t1, i.e. the first picts_2gates. VALUES IN SECONDS!\n
    beta: defined as t2/t1. t2 vcalues are obtained from this and t1\n
    t_avg: number of points to be averaged around t1 and t2. Not relevant if integrate=True. E.g. if t_avg=2, I average between i(t1) and the 2 points below and above, 5 in total. Same for i(t2).\n
    integrate: whether to perform double boxcar integration, i.e. calculating the integral of the current between t1 and t2 for each temperature (ref: Suppl. info of https://doi.org/10.1002/aenm.202003968 )\n
    round_en: integer indicating how many decimals the rate windows should should be rounded to. If None, the default calculated values of en are kept.\n
    injection: can be either "high" (default) or "low", corresponding to high or low injection from the light source. The expression for finding en is different in the 2 cases. \n

    Returns a dataframe with PICTS spectra and t2 values
    '''
    # Initial checks
    if (type(t1)!=np.ndarray and t1 is not None):
        raise TypeError('t1 must be numpy.ndarray object')
    # Create t1 values, if needed
    if t1 is None:
        if (t1_min==None or t1_shift==None or n_windows==None): raise ValueError("If t1 is not specified, you need to specify t1_min, t1_shift and n_windows")
        t1 = create_t1_values(t1_min, t1_shift, n_windows, method=t1_method)
    if (t1>tr.index.max()).any():      # If any value in t1 is bigger than the maximum time of the transients
        raise ValueError('Some t1 values are bigger than the highest value of the transient time index')

    # Create t2 based on t1 and beta
    t2 = np.array([beta*t1 for t1 in t1])
    if (t2>tr.index.max()).any():      # If any value in t2 is bigger than the maximum time of the transients
        raise ValueError('Some t2 values are bigger than the highest value of the transient time index. Adjust your t1 and beta accordingly.')

    t1_loc = np.array([tr.index.get_loc(t, method = 'backfill') for t in t1])    # location of t1 values. needed for using iloc later since loc has problems with tolerance
    t2_loc = np.array([tr.index.get_loc(t, method = 'backfill') for t in t2])    # location of t2 vcalues
    # Calculate rate windows
    #en = np.log(beta)/(t1*(beta-1))
    en = calculate_en(t1 = t1, t2 = beta*t1, injection=injection)
    # Calculate picts signal for each rate window taking the average of the current around t1 and t2 based on t_avg
    if integrate:
        picts = pd.concat([tr.iloc[t1:t2].apply(lambda x: scipy.integrate.trapz(x, tr.iloc[t1:t2].index)) \
                           for t1,t2 in zip(t1_loc,t2_loc) ], axis=1)
    else:
        picts = pd.concat([tr.iloc[t1-t_avg:t1+t_avg].mean() - tr.iloc[t2-t_avg:t2+t_avg].mean() \
                           for t1,t2 in zip(t1_loc,t2_loc)], axis=1)
    picts = round_rate_window_values(picts, en, round_en)
    picts.columns.name = 'Rate Window (Hz)'
    gates = np.array([t1, t2]).T       # I traspose it so that each row corresponds to a rate window
    
    return picts, gates



def picts_4gates (tr, alpha, beta, t_avg, t1_min=None, t1_method='linear', 
                  t1_shift=None, n_windows=None, gamma=None, t1=None, t4=None, 
                  integrate = False, round_en = None):
    '''
    tr: dataframe with transients at different temperatures\n
    t1_min: minimum value of t1. Necessary if t1 is not provided\n
    t1_shift: for the linear method it represents the "delta t" between one gate and the next
    n_windows: nuber of rate windows, i.e. number of t1 values of the returned array \n
    t1_method: method used to create t1 values. Accepted options are 'linear' (linearly increase the t1 values)
    t1: numpy array of values of t0, i.e. the first gates. VALUES IN SECONDS!\n
    t4: numpy array of values of t3, i.e. the last gates. Remember, the best is t4>9*t1\n
    alpha: defined as t2/t1. t2 values are obtained from this and t1\n
    beta: defined as t3/t1. t3 values are obtained from this and t1\n
    gamma: defined as t4/t1. t4 values are obtained from this if t4 is not explicitly passed.\n
    t_avg: number of points tobe averaged around the gates. Not relevant if integrate=True. E.g. if t_avg=2, I average between i(t1) and the 2 points below and above, 5 in total. Same for i(t2), i(t3), i(t4).\n
    integrate: whether to perform 4 gate integration, i.e. calculating the integral of the current between t2 and t3 divided by the same integral between t1 and t4 for each temperature (ref: Suppl. info of https://doi.org/10.1002/aenm.202003968 )
    round_en: integer indicating how many decimals the rate windows should should be rounded to. If None, the default calculated values of en are kept.

    Returns:
    1. a dataframe with PICTS spectra
    2. a numpy array with rate windows on rows and t1, t2, t3, t4 values on columns
    '''
    # Initial checks
    if (type(t1)!=np.ndarray and t1 is not None):
        raise TypeError('t1 must be numpy.ndarray object')
    if (alpha==beta):
        raise ValueError("alpha and beta have the same value, please set two different values for calculating the 4 gates spectrum.")
    if t1 is None:
        if (t1_min==None or t1_shift==None or n_windows==None): 
            raise ValueError("If t1 is not specified, you need to specify t1_min, t1_shift and n_windows")
        t1 = create_t1_values(t1_min, t1_shift, n_windows, method=t1_method)
    # Create t2 and t3 and t4 based on t1, alpha, beta, gamma
    t2 = t1*alpha
    t3 = t1*beta
    if t4 is None:        # Create t4 if not explicitly passed
        if gamma is None: raise ValueError("If t4 is not specified, you need to specify gamma")
        t4 = t1*gamma
    if (t4<10*t1).any():
        warnings.warn('Some or all t4 values are less than 10*t1, which is an essential condition for performing 4gates PICTS. Please, change them accordingly.')

    gates = np.array([t1, t2, t3, t4]).T       # I traspose it so that each row corresponds to a rate window
    # Check that no gate exceeds the maximum time index of the data
    for i,t in enumerate(gates):
        if (t>tr.index.max()).any():      # If any value in t is bigger than the maximum time of the transients
            raise ValueError(f"These gate values in this list are bigger than the highest value of the transient time index:\n {t} \n Adjust the input parameters accordingly")
    # Find index location of the gates so that later we can use iloc for defining time ranges
    gates_loc = np.array([np.array([tr.index.get_loc(t, method='backfill') for t in gate]) for gate in gates])
    # Calculate rate windows
    en = np.array([np.log((t[2]-t[0])/(t[1]-t[0])) / (t[2]-t[1]) for t in gates])

    # Calculate picts signal for each rate window taking the average of the current around t1 and t2 based on t_avg
    if integrate:
        # PICTS signal is the ratio of the current integral between t1 and t2 and the same integral between t0 and t3
        picts = pd.concat([tr.iloc[t[1]:t[2]].apply(lambda x: scipy.integrate.trapz(x, tr.iloc[t[1]:t[2]].index)) / \
                           tr.iloc[t[0]:t[3]].apply(lambda x: scipy.integrate.trapz(x, tr.iloc[t[0]:t[3]].index))\
                           for t in gates_loc ], axis=1)
    else:
        # Take the average of the currents in correspondence of the gates
        i_mean = [[tr.iloc[t-t_avg:t+t_avg].mean() for t in gate_loc] for gate_loc in gates_loc]
        # PICTS signal is the ratio of the current at t1 minus that at t2 and the same between t0 and t3
        picts = pd.concat([(i[1]-i[2])/(i[0]-i[3]) for i in i_mean], axis=1)
    picts = round_rate_window_values(picts, en, round_en)
    picts.columns.name = 'Rate Window (Hz)'

    return picts, gates




def picts_map (tr, t1_min, beta, t_avg, t1_shift, n_windows = None,
               alpha=None, gamma=None, integrate = False, round_en = None, 
               t1_shift_method='exp', injection = 'high'):
    '''
    tr: dataframe with transients at different temperatures\n
    t1_min: minimum value of t1. Necessary if t1 is not provided\n
    beta: For 2 gates defined as t2/t1. For 4 gates defined as t3/t1\n
    t_avg: number of points to be averaged around t1 and t2. Not relevant if integrate=True. E.g. if t_avg=2, I average between i(t1) and the 2 points below and above, 5 in total. Same for i(t2).\n
    t1_shift: It represents the "delta t" between one gate and the next. Its use depends on which method is used (See create_t1_values() for reference)
    n_windows: nuber of rate windows, i.e. number of t1 values of the returned array. If None, the max number of rate windows for the given t1_shift is generated \n
    alpha: Used for 4 gates method and defined as t2/t1. If passed, the picts method will be 4 gates.
    gamma: defined as t4/t1. If passed, the picts method will be 4 gates.\n
    integrate: whether to perform double boxcar integration, i.e. calculating the integral of the current between t1 and t2 for each temperature (ref: Suppl. info of https://doi.org/10.1002/aenm.202003968 )\n
    t1_shift_method: method used to create t1 values. Currently accepted options are 'exp' and 'linear' (linearly increase the t1 values)
    round_en: integer indicating how many decimals the rate windows should should be rounded to. If None, the default calculated values of en are kept.\n
    injection: can be either "high" (default) or "low", corresponding to high or low injection from the light source. The expression for finding en in the case of 2 gates PICTS is different in the 2 cases. \n

    Calculates the "3D Arrhenius plot", i.e. the PICTS spectrum over a large number of rate windows
    '''
    
    # Determine if the user wants 4 gates or 2 gates method
    if alpha is not None and gamma is not None:
        method = '4 gates'
        multiplier = gamma       # Used to determine the max value that t1 can have so that t4 is not larger than t_max
    else: 
        method = '2 gates'
        multiplier = beta       # Used to determine the max value that t2 can have so that t4 is not larger than t_max
    
    # In case n_windows was not passed to the function, create automatically t1 gates 
    if n_windows is None:
        t1=create_t1_values (tr = tr, t1_min = t1_min, multiplier = multiplier, t1_shift = t1_shift, method = t1_shift_method)
    # Create t1 gates in case n_windows was passed to the function
    else:
        if t1_shift is None: raise ValueError("You need to specify t1_shift if you specify n_windows")
        t1 = np.array([t1_min*np.exp(t1_shift*i) for i in range(n_windows)])
        t1 = t1[multiplier*t1<tr.index.max()]    # Make sure that t2 does not overcome the maximum time of the transient
    
    # Calculate PICTS spectrum and prepare the S dataframe to be converted into a xr.DataArray
    if method == '2 gates':
        S, gates = picts_2gates(tr, t1=t1, beta=beta, t_avg=t_avg,
                                integrate=integrate, injection = injection)
    if method == '4 gates':
        S, gates = picts_4gates(tr, t1=t1, alpha=alpha, beta=beta, gamma=gamma, 
                                   t_avg=t_avg, integrate=integrate)
        
    S_stack = S/S.max()     # Normalize all spectra to the max so that they have the same intensity
    S_stack.columns = np.log(S.columns)    # Convert en to log(en) for better visualization
    S_stack.index = 1000/S_stack.index     # Convert T to 1000/T
    S_stack=S_stack.stack().swaplevel()    #stack index and swap T and en indices. Operation needed to convert it into a DataArray easily
    S_stack.index=S_stack.index.set_names(['log_en', '1000/T'])    # Rename the indices
    
    # Create the xr.DataArray
    Sa = xr.DataArray.from_series(S_stack)
    Sa = Sa.sortby('log_en')
    Sa.name='Sa'                             # Set a name for the datarray, it will be needed by hvplot to produce a plotting of the data
    if round_en is not None: Sa['log_en'] = np.round(Sa['log_en'], round_en)                  # Round the log_en values. We do it here rather than when calling the picts method because those methods round the en value, not the log(en) one
    
    return Sa



###################################################################################
######## FITTING  #################################################################
###################################################################################




def gaus(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def gaus_fit (df, T_range, fit_window, return_Tm=False, outlier_window=None, outlier_threshold=None):
    '''
    df: Dataframe with different rate windows as columns, temperature as index.\n
    T_range: list-like. Temperature range where the peak is located.\n
    fit_window: Expressed in Kelvin. Since the peak positions move for different rate windows, for each rate windowthe peak is performed only in the range of +/- fit_window around the temperature at whichthe curve maximum is located. E.g. max is at 200K and fit_window=10, we just fit from 190Kto 210K.\n
    return_Tm: Whether to return also the position of the max of each spectrum, i.e. the Temperature at the max position (Tm). Useful when doing the arrhenius fit.
    outlier_window: int indicating window of the rolling median to find and remove the outliers before fitting (see remove_outliers function and https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/). If passed, you need to pass outlier_threshold as well. Increase this value if you have a large portion of your curves that contains outliers; if you have isolated spike you can use a low value. \n
    outlier_threshold: float or int indicating the variation on the values ("delta y") above which values are considered outliers. If passed, you need to pass outlier_window as well. The smaller, the more points will be excluded. (see remove_outliers function and https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/).\n
    Returns:
    a copy dataframe of df with gaussian fits

    '''
    # Check if index is monotonic, otherwise get_loc won't work
    if not df.index.is_monotonic:
        df = df.sort_index(ascending=True)        # Sort index ascendingly
        df = df[~df.index.duplicated(keep='first')]  # Delete rows with duplicates if there are any
        warnings.warn("The index of the dataframe is not monotonic, therefore the fitting function cannot work properly. I sorted the index and deleted duplicate index elements, please check that this operation is ok with your specific dataframe.",
        stacklevel=2)
        
    T_min_loc = df.index.get_loc(T_range[0], method='pad')    # Find position of Tmin and Tmax with tolerance for using iloc later
    T_max_loc = df.index.get_loc(T_range[1], method='backfill')
    df = df.iloc[T_min_loc: T_max_loc+1]                             # Restrinct dataframe only to desidred T range
    if outlier_window is not None and outlier_threshold is not None:
        df = remove_outliers(df, outlier_window, outlier_threshold)
    if len(df)==0:
        warnings.warn("Temperature range not correct. Try to switch the two temperature values in T_range.", stacklevel=2)
    max_pos = df.idxmax()                                          # Rough positions of picts maxima
    df_fits = []                                                   # list containing fitted dataframes
    if return_Tm: Tm =  pd.Series(index=df.columns, dtype='float64')     # Create series that will contain the Tm values with rate windows as index
    for i,rate in enumerate(df):                                         # loop thorugh rate windows
        T_min_fit = max(max_pos[rate] - fit_window, T_range[0])          # T min considering fit window. If the result is lower than the global T_min, we consider T_min instead
        T_max_fit = min(max_pos[rate] + fit_window, T_range[1])          # Same as for T_min
        T_min_fit_loc = df.index.get_loc(T_min_fit, method='backfill')   # location of the previous values for use of iloc
        T_max_fit_loc = df.index.get_loc(T_max_fit, method='backfill')
        # Fitting
        y = df.iloc[T_min_fit_loc:T_max_fit_loc, i].dropna().values
        x = df.iloc[T_min_fit_loc:T_max_fit_loc, i].dropna().index.values
        A_guess = y.max()                                                 # Amplitude guess is curve max
        x0_guess = max_pos[rate]                                          # T max guess is the idxmax of the dataframe
        sigma_guess = x.std()                                             # sigma guess is the standard deviation of temperatures in this range
        popt, pcov = curve_fit(gaus, x, y, p0=[A_guess,x0_guess,sigma_guess] )     # Do gaussian fit
        # Save results
        if return_Tm: Tm[rate] = popt[1]
        x_values = df.iloc[T_min_fit_loc:T_max_fit_loc, i].index.values     # We cannot use x because if dropna() removed any values, then the fit will not be a continuous curve, but will contain NaN values
        df_fits.append(pd.DataFrame(index = x_values, data = gaus(x_values, *popt)))
    df_fit = pd.concat(df_fits, axis = 1)                                   # Concatenate all results in a single dataframe
    df_fit.index.name = df.index.name
    df_fit.columns = df.columns
    df_fit.columns.name = df.columns.name
    if return_Tm: return df_fit, Tm
    else: return df_fit




def arrhenius_fit (S, T_traps, fit_window, m_eff_rel, exclude_en=[], outlier_window=None, outlier_threshold=None):
    '''
    S: dataframe with PICTS signal (columns are rate windows, index is temperature). Alternatively, can be a tuple outputed by picts_2gates/picts_4gates, where the first element is the picts DataFrame.\n
    T_traps: dictionary where the key is the trap name and the value is a list of 2 values indicating the temperature range whewre the corresponding peaks appear\n
    fit_window: Expressed in Kelvin. Can be either int/float value or dictionary with keys corresponding to the Trap names expressed in T_traps.
                If it's a int/float, the same fit window will be used for all traps
                Since the peak positions move for different rate windows, for each rate window
                the fit is performed only in the range of +/- fit_window around the temperature at which
                the curve maximum is located. E.g. max of a rate window is found at 200K and fit_window=10, then we just fit from 190K
                to 210K.\n
    m_eff_rel: relative effective mass i.e. the dimensionless quantity m_eff/m_e, where m_e is the electronic mass.\n
    exclude_en: list of int or dictionary of lists of int. The list indicates which rate windows not to consider in the gaussian fit of the peaks. Integers should be numbers from 0 to n_windows-1. 
                It can also be a dictionary where keys must be the same as the ones of T_traps
                and the values should be lists indicating the ens to exclude for each trap.
    outlier_window: int indicating window of the rolling median to find and remove the outliers before fitting (see remove_outliers function and https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/). If passed, you need to pass outlier_threshold as well. Increase this value if you have a large portion of your curves that contains outliers; if you have isolated spike you can use a low value. \n
    outlier_threshold: float or int indicating threshold above which points are considered outliers. If passed, you need to pass outlier_window as well. The smaller, the more points will be excluded. (see remove_outliers function and https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/).\n

    Returns:
    0 a dataframe with arrhenius plot data,
    1 a dataframe with arrhenius plot fits
    2 a dataframe with the gaussian fits of the picts spectrum for each trap
    3 a dataframe with trap parameters (Ea,sigma)
    '''
    # If fit_window and exclude_en are not dictionaries, we create a dictionary where the values for each trap are the same as the one passed by the user 
    if isinstance(fit_window, (int,float)):
        fit_window = dict.fromkeys(T_traps.copy(),fit_window)    # Create a copy of T_traps dict so that we have the same keys, then set value equal to fit_window for all keys using dict.fromkeys()
    if isinstance(exclude_en, list):
        exclude_en = dict.fromkeys(T_traps.copy(),exclude_en)    # Same as for fit_window
    # If a tuple from picts_2gates is passed we exctract just the first item: the DataFrame
    if isinstance(S, tuple):
        S=S[0]
    # Gaussian peak fitting for finding Tm, temperature corresponding to peak max
    fits = {}    # Dictionary that will be filled with dataframes with gaussian fits, one for each trap
    Tms = {}     # Dictionary that will be filled with dataframes with Tm, one for each trap
    for trap in T_traps:
        S_to_fit = S.drop(S.columns[exclude_en[trap]], axis=1)       # Before fitting, drop the columns that the user wants to exclude. I create a copy of S (S_to_fit), otherwise I loose some data on my original dataframe
        fits[trap], Tms[trap] = gaus_fit(S_to_fit, T_range = T_traps[trap], 
                                         fit_window=fit_window[trap], return_Tm=True,
                                         outlier_window=outlier_window, outlier_threshold=outlier_threshold)  # do gaussian fit of all peaks of a certain trap
    S_fit = pd.concat(fits, axis=1)                                       # multi-column dataframe with gaussian fits. First level are the traps, second level are the rate windows for each trap
    Tm = pd.concat(Tms, axis=1)
    # Create arrhenius plot DataFrame
    en = Tm.index.values    # extract rate windows from Tm index
    arrhenius = pd.concat([pd.DataFrame(data = np.log((Tm**2)[trap]/en).values,
                                        index = 1000/Tm[trap].values,
                                        columns = [trap])
                           for trap in Tm.columns])
    #arrhenius.columns.name = 'ln(T²/en)'
    arrhenius.index.name='1000/T (K⁻¹)'

    # Linear fit of arrhenius plots
    trap_params = pd.DataFrame(columns=arrhenius.columns.values)    # Dataframe containing sigma and Ea for each trap
    trap_params.columns.name=''
    m_eff = m_eff_rel * m_e
    gamma = 2 * np.sqrt(3) * (2*np.pi)**(3/2) * k_B**2 * m_eff * h**(-3)    # See Orton-Blood Equation 7.19, page 344: en(T) = gamma * T^2 * sigma * exp(-Ea/(kT))

    # Fit the data
    arrhenius_fits = {}           # Dictionary of dataframes with fits of each single arrhenius plot
    for trap in arrhenius:
        x=arrhenius[trap].dropna().index
        y = arrhenius[trap].dropna().values
        (slope, interc), cov = np.polyfit(deg=1, cov=True, x=x, y=y)
        # Extract Ea and sigma
        trap_params.loc['Eₐ (eV)', trap]=slope*k_B*1000/q    # Activation energy in eV (since slope is = Ea/(1000*k_B), + conversion from J to eV by "q")
        trap_params.loc['δEₐ (eV)', trap] = np.sqrt(cov.diagonal())[0]*k_B*1000/q      # Error on Ea, calculated as sqrt of the slope diagonal covalence matrix element, then scaled by the same constants as Ea
        trap_params.loc['σ (cm²)', trap] = np.exp(-interc)/gamma*1e4   # Capture cross section in cm^2 (since interc = -ln(sigma*gamma) + conversion from m2 to cm2)
        trap_params.loc['δσ (cm²)', trap] = trap_params.loc['σ (cm²)', trap] * np.sqrt(cov.diagonal()[1])    # Error on sigma, calculated by error propagation for an exponential function
        # Build linear fit curve
        y_fit = interc + slope*x
        arrhenius_fits[trap] = pd.DataFrame(data = y_fit, index = x)
        trap_params.loc['r²', trap] = r2_score(y_true=y, y_pred=y_fit)
        if not arrhenius_fits[trap].index.is_unique:
            warnings.warn('Some Temperature values of '+trap+"'s Arrhenius plot are duplicated, so I'm dropping the duplicated ones, keeping just the first.", stacklevel=2)
            arrhenius_fits[trap] = arrhenius_fits[trap][~arrhenius_fits[trap].index.duplicated(keep='first')]
    arrhenius_fit = pd.concat(arrhenius_fits, axis=1)
    arrhenius_fit.columns = arrhenius.columns.values

    return arrhenius, arrhenius_fit, S_fit, trap_params



def map_fit (S, T_range, log_en_range, n_points, fit_window, m_eff_rel, exclude_en = None, outlier_window=None, outlier_threshold=None):
    '''
    S: xr.DataArray returned by picts_map(), with 1000/T and log_en as coords and PICTS signal as values.\n\n
    T_range: dictionary where the key is the trap name and the value is a list of 2 values indicating the temperature range (in Kelvin) whewre the corresponding peaks appear\n\n
    log_en_range: list or dict of lists containing min and max log_en values among which the trap is located. If dict, the keys have to be the same trap names in T_range\n\n
    n_points: int or dict of ints which represent the desired number of points of the arrhenius plot. If dict, the keys have to be the same trap names in T_range\n\n
    fit_window: same as for arrhenius_fit()\n\n
    m_eff_rel: relative effective mass of the material\n\n
    exclude_en = list of int or dictionary of lists of int containing the positions of the rate windows not to be considered in the fit (see arrhenius_fit for reference)
    '''
    # If log_en_range is not a dictionary, we create a dictionary where the values for each trap are the same as the one passed by the user via T_range
    if isinstance(log_en_range, list):
        log_en_range = dict.fromkeys(T_range.copy(),log_en_range)  
    if isinstance(n_points, int):
        n_points = dict.fromkeys(T_range.copy(),n_points)
    if isinstance(exclude_en, list):
        exclude_en = dict.fromkeys(T_range.copy(),exclude_en) 
    
    arrs = {}    # list of arrhenius fits
    en_min_spacing = np.diff(S.coords['log_en'].values)[1]     # Minimum spacing that can be achieved given the en resolution of the array
    for trap in T_range:
        en_spacing = (log_en_range[trap][1]-log_en_range[trap][0]) / n_points[trap]          # spacing between two consecutive en_values, given the resolution requested by the user
        if en_spacing<en_min_spacing: raise ValueError("There is not enough resolution on the log_en axis. Decrease the n_points or generate a map with smaller en steps by decreasing t1_shift")    
        en_values = np.arange(log_en_range[trap][0], log_en_range[trap][1], en_spacing)  # en values corresponding to the horizontal profiles to be extracted from the array for fitting
        # Create the subset of the big array with the profiles to be fitted and convert it to pd.DataFrame since this is what arrhenius_fit() needs.
        s = S.sel(log_en = en_values, method='bfill').to_dataframe().unstack().T.droplevel(0)   
        # We exclude the rate windows here instead of inside arrhenius_fit because in this way we can keep track of the un-fitted columns, which are added back as nan columns a few lines below
        s.index = 1000/s.index          # arrhenius fit works with T values
        s = s.sort_index()              # sort index to avoid errors during arrhenius fitting
        if exclude_en is not None: s_to_fit = s.drop(exclude_en[trap], axis=1)
        else: s_to_fit = s.copy()
        log_en_values = s_to_fit.columns                    # Save the log(en) values as in the next line we switch to en and later we want to go back to log(en). Note, doing again log on the columns to go back to log(en) can create problems with some rate window values. E.g. log(exp(0.637))=0.636999999, which then gives problems in the code.
        s_to_fit.columns = np.exp(s_to_fit.columns.values)   # arrhenius fit works with en values      
        try:
            # Arrhenius fit
            arr = arrhenius_fit(s_to_fit, T_traps= {trap: T_range[trap]},
                                fit_window = fit_window,
                                m_eff_rel = m_eff_rel, 
                                outlier_window=outlier_window, outlier_threshold=outlier_threshold
                         )
        except RuntimeError: 
            print("The gaussian fitting of the peaks failed for trap %s. These are the spectra you gave as input as well and the temperature range values as vertical blue lines:\n" % (trap,))
            plt.plot(s_to_fit)
            plt.axvline(x=T_range[trap][0], c='b', linestyle='--')
            plt.axvline(x=T_range[trap][1], c='b', linestyle='--')
            plt.show()
            raise ValueError("The gaussian fitting failed.")
        
        # Adjust the s_fit output
        s_fit = arr[2]
        s_fit.columns = s_fit.columns.set_levels(log_en_values, level=1) # Go back to log(en)
        # Add back the un-fitted columns so that when we call plot_all we can also see which columns were not fitted
        missing_en = list(set(s.columns)-set(s_fit[trap].columns))
        
        # Add the missing rate windows as nan columns. 
        # However, it can happen that the operation of transforming in exp and then back to log does not yield the same rate window. Un example is np.log(np.exp(0.637))=0.63699999... 
        # Therefore we check that the missing en is not very close to one of the existing ones(difference<1e-10), in which case we 
        cols = s_fit.columns.levels[1]
        for en in missing_en: s_fit.loc[:,(trap,en)]=np.nan    
        
        s_fit.sort_index(axis=1, inplace=True)    # Sort column index so to have all rate windows in ascending order, as the excluded ones are added at the end of the df, not in order

        # Update the dictionary with all arrhenius outputs for each trap
        arrs[trap] = arr
    # make a tuple that contains (arr, arr_fit, S_fit, trap_params) where these objects are dataframes with multiindex columns whose first level is the trap name
    arr = tuple([pd.concat([ i[j] for i in list(arrs.values()) ]) for j in range(3)])  # arr, arr_fit, S_fit are concatemated automatically on the 1 axis
    arr += (pd.concat( [i[3] for i in list(arrs.values())], axis=1 ),)                 # for trap_params we have to explicitly concat on axis=1, so we do it separately and append it to the final tuple
    return arr


################################################################################


################################################################################
### PLOTTING + VISUALIZATION ###################################################
################################################################################

def plot_transients (tr, gates = None, cmap=None, **hvplot_opts):
    '''
    Plots the transients with an interactive widget allowing to visualize different temperatures. Returns an hvplot object.\n
    tr: Dataframe with time on index (default name 'Time (s)') and temperatures on columns (default name 'Temperature (K)')\n
    gates: numpy array containing rate windows in rows and t1,t2, and optionally t3 and t4, in columns (as returned by picts_2gates and picts_4gates)
    cmap: colormap for the spectrum
    hvplot_opts: hvplot parameters to customize the spectrum plot.
    '''
    # Default options
    opts = dict(x='Time (s)', y=0, width=700,
                ylabel = 'Current (A)', color = 'k')
    # Overwrite or add the user specified options to the otpions used to produce the plot
    for opt in hvplot_opts: opts[opt] = hvplot_opts[opt]
    
    # Overlay rate window visualization if specified
    if gates is not None:
        
        # Define colormap
        if cmap is None: colormap = hv.Cycle.default_cycles["default_colors"]
        else: colormap = cm(gates.shape[0], cmap)

        # Vertical lines corresponding to gate positions
        lines = hv.Overlay([hv.Overlay([hv.VLine(x=ti).opts(color=colormap[i]) for ti in t]) \
                            for i,t in enumerate(gates)])
            
        # Return a pn.interact where the function is a lambda that returns an Overlay of tr and VLines and the variable is Temperature
        return pn.interact(lambda Temperature: tr[Temperature].hvplot(**opts)*lines,
                           Temperature=list((tr.columns)))
    else: 
        return pn.interact(lambda Temperature: tr[Temperature].hvplot(**opts),
                           Temperature=list((tr.columns)))


    
    
def plot_arrhenius(arr, arr_fit=None, suffix='', **hvplot_opts):
    '''
    Plots the arrhenius data with the correct x and y labels
    arr: Series, Dataframe or tuple of DataFrames (as returned by arrhenius_fit) with 1000/T as index and ln(en/T^2) values on the columns. Each column should represent a different trap
    arr_fit: Same structure as arr, should contain the fit of the corresponding arrhenius lines.
    suffix: string with text to be added to all the legend items. Useful when doing overlays of different plot_arrhenius() outputs with the same labels (e.g. two consecutive scans where one sees the same trap)
    hvplot_opts: options to be passed to hvplot plotting library
    '''
    if isinstance(arr, tuple): arr, arr_fit = arr[0].copy(), arr[1].copy()
    if isinstance(arr, pd.Series): arr = arr.to_frame()
    # Default plotting options
    opts = dict(kind='scatter', ylabel='ln(T²/en)', xlabel='1000/T (K⁻¹)',
                hover_cols=['T (K)', 'en (Hz)', 'ln(en)'])
    # Overright or add options, if needed
    for key, value in hvplot_opts.items(): opts[key]=value
    if suffix!= '':
        arr.columns += suffix      # add suffix to the arr columns
        arr_fit.columns += suffix      # add suffix to the arr_fit columns
    traps = list(arr.columns)
    # Create new columns for hover info (temperature, en, log(en) of each point)
    arr['T (K)'] = 1000/arr.index
    arr['en (Hz)'] = arr['T (K)']**2/np.exp(pd.concat([arr[trap] for trap in traps]).dropna().values)
    arr['ln(en)'] = np.log(arr['en (Hz)'])
    plot = arr.hvplot(y=traps, **opts)
    if arr_fit is not None:
        plot = plot*arr_fit.hvplot(color='red', hover=False)
    return plot



def plot_map (S, interact=False, **hvplot_opts):
    
    '''
    S: xr.DataArray returned by picts_map(), with 1000/T and log_en as coords and PICTS signal as values.\n\n
    interact: whether to display also a graph with horizontal map profile that updates when the mouse moves on the map. \n\n 
    hvplot_opts: map plotting options to be passed to the hvplot constructor
    '''
    
    opts = dict(cmap='turbo', rasterize=True, width=400, 
                ylabel = 'ln(en)', xlabel='1000/T (K^-1)')
    # Update the default opts if the user has specified different ones
    opts.update(hvplot_opts)
    
    if interact:
        qmesh = S.hvplot.quadmesh(dynamic=False, **opts)   # Create 3D plot
        # Create a copy dataarray with T coord instead of 1000/T and create a new 3D plot based on this
        # We will use this to display the y profile as a function of T instead of 1000/T, since the fitting will be done on the spectrum vs T and not vs 1000/T
        S_T = S.copy()
        S_T.coords['1000/T']=1000/S.coords['1000/T']
        S_T = S_T.rename({'1000/T': 'T'})
        qmesh_T=S_T.hvplot.quadmesh(dynamic=False, **opts)

        # Define pointer that tracks the mouse position over the qmesh map
        pointer = streams.PointerXY(x=S.coords['1000/T'].values.min(), 
                            y=S.coords['log_en'].values.min(),
                            source=qmesh)
        # Define the horizontal profile plot and link it to the pointer
        y_sample = hv.DynamicMap(lambda x, y: qmesh_T.apply.sample(log_en=y), 
                                 streams=[pointer])
        # Options for y_sample plot
        y_sample_opts = dict(height=300, width=400, tools=['hover'],
                             ylabel='PICTS signal (a.u.)', xlabel='Temperature (K)',
                             ylim=(S.values.min(), S.values.max()))
        out = qmesh + y_sample.opts(**y_sample_opts)
    else:
        out = S.hvplot.quadmesh(**opts)
    
    return out



def plot_all(picts, arrhenius=None, tr=None, show_arrhenius=True, **S_plot_opts):
    '''
    picts: tuple object returned by picts_2gates/picts_4gates, or dataframe containing the spectra, or xr.DataArray containing the 3D Arrhenius.\n
    tr: transients for rate window visualization.\n
    arrhenius: tuple object returned by arrhenius_fit.\n
    show_arrhenius: whether to show the arrhenius plot or no.\n
    S_plot_opts: additional hvplot () parameters for the spectrum plot
    '''
    # Plot options
    S_opts = dict(width=400, height=350, color='black',alpha=0.5,legend=False)
    S_opts.update(S_plot_opts)
    S_fit_opts = dict(line_width=3, legend=False)
    tr_opts = dict(width=400,height=300,
                   xlim=[0,None])
    arr_opts = dict(height=350, width=250, legend=False)
    
    if arrhenius is None:
        show_arrhenius=False
    else:  # Get all components from arrhenius data
        arr, arr_fit, S_fit, trap_params = arrhenius
        
    # Get spectrum and gates from picts data
    if isinstance(picts, tuple):    # tuple object returned by picts_2gates/picts_4gates             
        S = picts[0]   
        gates = picts[1]
    elif isinstance(picts, xr.DataArray):     # If it's a 3D Arrhenius map returned by picts_map
        if show_arrhenius == False:
            return plot_map(picts)        
        else:
            # Select from the map only the fitted profiles, based on the column names of the arrhenius fit dataframe
            S = picts.sel(log_en=S_fit.columns.levels[1], method='ffill').to_dataframe().unstack(0).droplevel(0,axis=1)
            S.index = 1000/S.index     # Change the index to 1000/T
            S.index.name = 'Temperature (K)'
    else: S = picts                # Case of simple dataframe with picts spectra
    
    ### LEFT COLUMN ###    
    # Create the spectrum plot and put it in a column that will be on the left of the final panel
    if show_arrhenius==False: 
        plot_S = S.hvplot(**S_opts)    # Show lines if the gauss fit is not overlayed
    else: 
        ## If the user used exclude_en there will be some en_values missing and calling S_fit[en] below will fail. We want the plot to show no fit if the specified en was excluded, so we want to add new NaN columns for these ens
        ## The other case when we want to do this is when more than 1 traps are present and the two traps are not fitted on the same en values. In that case S_fit.columns will be bigger than S_fit.columns.levels[0]
        if len(S.columns) != len(S_fit.columns.levels[1]) or len(S_fit.columns) > len(S_fit.columns.levels[0]):
            for trap in S_fit.columns.levels[0]:
                missing_en = list(set(S.columns)-set(S_fit[trap].columns))     # Find which en values are missing
                for en in missing_en: S_fit.loc[:,(trap,en)]=np.nan            # Create new column with nan vlaues and correct column name
        # Define plot color for each trap so that they are easily distinguishable when plotting
        cmap = [c for c in mcolors.TABLEAU_COLORS.values()]
        #cmap.pop('tab:red')             # Remove red color which is the same as for the linear fit
        gaus_fit_colors = {trap: cmap[i] for i, trap in enumerate(S_fit.columns.levels[0])} 
        plot_S = pn.interact(lambda ln_en: hv.Overlay(S[ln_en].hvplot(kind='scatter', size=1, 
                                                                title='ln(en) = ' + str(ln_en) + '    en = '+str(np.round(np.exp(ln_en),3))+' Hz',
                                                                **S_opts)*\
                                                   hv.Overlay([S_fit[trap][ln_en].hvplot(color=gaus_fit_colors[trap],**S_fit_opts) for trap in arr.columns])),
                             ln_en = S.columns
                            )
    left_col = pn.Column(plot_S)
    
    # If the transients are provided, create the plot and append it to the left column
    if tr is not None:
        plot_tr = plot_transients(tr,gates=gates, **tr_opts)
        left_col.append(plot_tr)    
    
    panel = pn.Row(left_col)
    
    ### RIGHT COLUMN ###    
    if arrhenius is not None and show_arrhenius==True:
        plot_arr = hv.Layout([plot_arrhenius(arr[trap].dropna(), arr_fit[trap].dropna(), 
                                             title=trap, color=gaus_fit_colors[trap], **arr_opts)\
                              .opts(show_legend=False, axiswise=True) \
                              for trap in arr])
        
        right_col= pn.Column(plot_arr, trap_params.interactive())
        panel.append(right_col)
    
    return panel