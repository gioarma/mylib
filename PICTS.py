from nptdms import TdmsFile
import pandas as pd
import numpy as np
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
from mylib.plotting import cmap as cm
import hvplot.pandas
import holoviews as hv
import panel as pn
from datetime import datetime

###### Physical constants   #######################
k_B = physical_constants['Boltzmann constant'][0]
q = physical_constants['elementary charge'][0]
h = physical_constants['Planck constant'][0]
m_e = physical_constants['electron mass'][0]
###################################################



####### DATA IMPORT/EXPORT  #######################################################################

def read_transients (path, amplifier_gain, dropna=False, set_timetrack = True, drop=None):

    '''
    path: string with file path of TDMS file
    amplifier_gain: gain set on the current amplifier (to extract current from voltage values)
    current_scale: Order of magnitude of the currents to scale the data (e.g. 1e-9 sets values to nA). Set 1 to leave as measured
    time_scale: Order of magnitude of the times to scale the data (e.g. 1e-6 sets values to µA). Set 1 to leave as measured
    dropna: whether to drop all rows where there is at least one NaN. Useful for faulty dataset where there are some columns with more data than others
    set_timetrack: whether to get the timetrack from the files. In some corrupted data it is better to avoid it
    drop: list of 2 integers indicating the initial and final columns to be dropped from the dataframe as soon as it is imported. Usually used to remove the first or last columns which may contain corrupted data. If drop=[0,5] drop the first 2 columns
    '''

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
    df.index+=trigger

    return df


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



def save_arrhenius (arr, trap_params, sample_name, picts_method , info = '', path = ''):
    
    '''
    Saves the arrhenius plots in a csv file for further analysis
    
    arr: DataFrame/list of DataFrames. Each DataFrame should contain the arrhenius plots in form of 1000/T as index, ln(T2/en) as columns
    sample_name = string containing the sample name (use just the code, not any pre-code like 'MAPbBr' or else)
    picts_method = string containing the method used to obtain the arrhenius plot. Can be: '2gates', '4gates', 'integral2gates', 'integral4gates'
    info = any additional information on the arrhenius plot, e.g. 'scan1', 'scan2', 'X-ray-irradiated', etc. Do not use underscores to separate words, use just dashes
    path: path where to save the data. If not specified, the csv is saved in the working directory
    '''
    
    ## Check that arr is either DataFrame or list ##
    if not isinstance(arr, (list, pd.DataFrame)):
        raise TypeError("The arr parameter should be either a DataFrame or a list of DataFrames")
    ## If arr is list, check that it contains DataFrames
    if isinstance(arr, list):
        for df in arr: 
            if not isinstance(df, pd.DataFrame):
                raise TypeError("arr should be a list of DataFrame objects only.")
            if len(arr) != len(trap_params):   # Check that arr and trap_params contain the same number of elements
                raise ValueError("arr list and trap_params lists do not have the same size.")
    ## Check that trap_params is either DataFrame or list ##
    if not isinstance(trap_params, (list, pd.DataFrame)):
        raise TypeError("The trap_params parameter should be either a DataFrame or a list of DataFrames")
    ## If trap_params is list, check that it contains DataFrames
    if isinstance(trap_params, list):
        for df in trap_params: 
            if not isinstance(df, pd.DataFrame):
                raise TypeError("trap_params should be a list of DataFrame objects only.")

    # If user diddn't put / at the end of path, we add it
    if path != '':
        if path[-1] != '/': path = path+'/'
    
    # If arr is a single df, we put it into a 1-element list 
    if isinstance(arr, pd.DataFrame):
        arr = [arr]
    if isinstance(trap_params, pd.DataFrame):
        trap_params = [trap_params]
    
    # Get current time expressed from year to seconds
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Create full path + file name
    if info=='': 
        arr_filename = path + 'arrhenius_' + sample_name + '_' + picts_method + '_' + current_time+'.csv'
    else: 
        arr_filename = path + 'arrhenius_' + sample_name + '_' + picts_method + '_' + info + '_' + current_time+'.csv'
        
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
    df['Sample name'] = sample_name
    df['PICTS method'] = picts_method
    df['Info'] = info
    # Save the csv file
    df.to_csv(arr_filename)
    

def import_arrhenius(path='arrhenius_data', sample_info_path='sample_info.xlsx'):
    
    '''
    Imports the arrhenius plot data, getting all the sample information from the file name and from the excel file with further info on the samples\n\n
    
    path: string containing the path to the folder where the csv files f the arrhenius plots are stored.\n\n
    sample_info_path: string containing the path to the excel file that contains all the additional info on the samples
    
    '''
    ## Get info from file name and create the dataframe
    
    file_list=natsorted(glob.glob(path+'/*'))  # Find all files in the folder
    ## Concatenate all data in a single DataFrame
    arr = pd.concat([pd.read_csv(f) for f in file_list ])
    arr['Sample name'] = arr['Sample name'].astype(str)      # convert sample name to string, since it can be seen as an integer by read_csv and create problems when joining dataframes afterwards
    # We convert this into a multiindex dataframe so that we can join it with sample_info after. We have to do this because arr and sample_info do not have the same amount of rows for each sample (sample_info has only 1 row for each sample)
    arr['count'] = arr.groupby('Sample name').cumcount()        # Create a count column, which is the inner level of the multiindex. It's simply a count of each repetition of Sample name
    arr.set_index(['Sample name','count'], inplace=True)        # The outer level of the multiindex is Sample name
    
    ## Get other info from the excel file
    sample_info = pd.read_excel('sample_info.xlsx', engine='openpyxl', index_col='Sample name')
    sample_info.index = sample_info.index.astype(str)       # On excel the sample name may be considered as a int instead of a string if there are no letters in the name, so we convert it to string
    
    df = arr.join(sample_info).reset_index().drop('count', axis=1)   # joining the two dfs, then resetting index and dropping count (I needed them just to perferm the join smoothly)
    df = df.rename(columns={'Unnamed: 0': 'Trap'})                   # Trap column was unnamed from read csv, so we give it a name here
    return df

###############################################################################################################





###### DATA ANALYSIS####################################################################

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


def round_rate_window_values (df, en, round_value):
    '''
    df: input dataframe where columns are supposed to be en values

    en: rate window values

    round_value: decimal position en windows should be rounded to

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



def picts_2gates (tr, t1, beta, t_avg, integrate = False, round_en = None, injection = 'high'):
    '''
    tr: dataframe with transients at different temperatures\n
    t1: numpy array of values of t1, i.e. the first picts_2gates. VALUES IN SECONDS!\n
    beta: defined as t2/t1. t2 vcalues are obtained from this and t1\n
    t_avg: number of points to be averaged around t1 and t2. Not relevant if integrate=True. E.g. if t_avg=2, I average between i(t1) and the 2 points below and above, 5 in total. Same for i(t2).\n
    integrate: whether to perform double boxcar integration, i.e. calculating the integral of the current between t1 and t2 for each temperature (ref: Suppl. info of https://doi.org/10.1002/aenm.202003968 )\n
    round_en: integer indicating how many decimals the rate windows should should be rounded to. If None, the default calculated values of en are kept.\n
    injection: can be either "high" (default) or "low", corresponding to high or low injection from the light source. The expression for finding en is different in the 2 cases. \n

    Returns a dataframe with PICTS spectra and t2 values
    '''
    # Initial checks
    if (type(t1)!=np.ndarray):
        raise TypeError('t1 must be numpy.ndarray object')
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

    return picts, t2



def picts_4gates (tr, t1, t4, alpha, beta, t_avg, integrate = False, round_en = None):
    '''
    tr: dataframe with transients at different temperatures\n
    t1: numpy array of values of t0, i.e. the first gates. VALUES IN SECONDS!\n
    t4: numpy array of values of t3, i.e. the last gates. Remember, the best is t4>9*t1\n
    alpha: defined as t2/t1. t2 values are obtained from this and t1\n
    beta: defined as t3/t1. t3 values are obtained from this and t1\n
    t_avg: number of points tobe averaged around the gates. Not relevant if integrate=True. E.g. if t_avg=2, I average between i(t1) and the 2 points below and above, 5 in total. Same for i(t2), i(t3), i(t4).\n
    integrate: whether to perform 4 gate integration, i.e. calculating the integral of the current between t2 and t3 divided by the same integral between t1 and t4 for each temperature (ref: Suppl. info of https://doi.org/10.1002/aenm.202003968 )
    round_en: integer indicating how many decimals the rate windows should should be rounded to. If None, the default calculated values of en are kept.

    Returns:
    1. a dataframe with PICTS spectra
    2. a numpy array with rate windows on rows and t1, t2, t3, t4 values on columns
    '''
    # Initial checks
    if (type(t1)!=np.ndarray):
        raise TypeError('t1 must be numpy.ndarray object')
    if (t4<10*t1).any():
        warnings.warn('Some or all t4 values are less than 10*t1, which is an essential condition for performing 4gates PICTS. Please, change them accordingly.')
    if (alpha==beta):
        raise ValueError("alpha and beta have the same value, please set two different values for calculating the 4 gates spectrum.")
    # Create t1, t2 and t3 based on t0 and beta
    t2 = t1*alpha
    t3 = t2*beta

    gates = np.array([t1, t2, t3, t4]).T       # I traspose it so that each row corresponds to a rate window
    # Check that no gate exceeds the maximum time index of the data
    for i,t in enumerate(gates):
        if (t>tr.index.max()).any():      # If any value in t is bigger than the maximum time of the transients
            raise ValueError(f"These t{i+1} values are bigger than the highest value of the transient time index:\n {t} \n Adjust the input parameters accordingly")
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



def gaus(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def gaus_fit (df, T_range, fit_window):
    '''
    df: Dataframe with different rate windows as columns, temperature as index

    T_range: list-like. Temperature range where the peak is located

    fit_window: Expressed in Kelvin. Since the peakpositions move for different rate windows, for each rate window
                the peak is performed only in the range of +/- fit_window around the temperature at which
                the curve maximum is located. E.g. max is at 200K and fit_window=10, we just fit from 190K
                to 210K

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
    if len(df)==0:
        warnings.warn("Temperature range not correct. Try to switch the two temperature values in T_range.", stacklevel=2)
    max_pos = df.idxmax()                                          # Rough positions of picts maxima
    df_fits = []                                                   # list containing fitted dataframes
    for i,rate in enumerate(df):                                         # loop thorugh rate windows
        T_min_fit = max(max_pos[rate] - fit_window, T_range[0])          # T min considering fit window. If the result is lower than the global T_min, we consider T_min instead
        T_max_fit = min(max_pos[rate] + fit_window, T_range[1])          # Same as for T_min
        T_min_fit_loc = df.index.get_loc(T_min_fit, method='backfill')   # location of the previous values for use of iloc
        T_max_fit_loc = df.index.get_loc(T_max_fit, method='backfill')
        y = df.iloc[T_min_fit_loc:T_max_fit_loc, i].values
        x = df.iloc[T_min_fit_loc:T_max_fit_loc, i].index.values
        A_guess = y.max()                                                 # Amplitude guess is curve max
        x0_guess = max_pos[rate]                                          # T max guess is the idxmax of the dataframe
        sigma_guess = x.std()                                             # sigma guess is the standard deviation of temperatures in this range
        popt, pcov = curve_fit(gaus, x, y, p0=[A_guess,x0_guess,sigma_guess] )     # Do gaussian fit
        df_fits.append(pd.DataFrame(index = x, data = gaus(x, *popt)))
    df_fit = pd.concat(df_fits, axis = 1)                                   # Concatenate all results in a single dataframe
    df_fit.index.name = df.index.name
    df_fit.columns = df.columns
    df_fit.columns.name = df.columns.name
    return df_fit


def arrhenius_fit (S, T_traps, fit_window, m_eff_rel):
    '''
    S: dataframe with PICTS signal (columns are rate windows, index is temperature)

    T_traps: dictionary where the key is the trap name and the value is a list of 2 values indicating the temperature range whewre the corresponding peaks appear

    fit_window: Expressed in Kelvin. Since the peak positions move for different rate windows, for each rate window
                the peak is performed only in the range of +/- fit_window around the temperature at which
                the curve maximum is located. E.g. max of a rate window is at 200K and fit_window=10, then we just fit from 190K
                to 210K

    m_eff_rel: relative effective mass i.e. the dimensionless quantity m_eff/m_e, where m_e is the electronic mass.

    Returns:
    1 a dataframe with arrhenius plot data,
    2 a dataframe with arrhenius plot fits
    3 a dataframe with the gaussian fits of the picts spectrum for each trap
    4 a dataframe with trap parameters (Ea,sigma)
    '''
    # Gaussian peak fitting for finding Tm, temperature corresponding to peak max
    fits = {}
    for trap in T_traps:
        fits[trap] = gaus_fit(S, T_range = T_traps[trap], fit_window=fit_window)  # do gaussian fit of all peaks of a certain trap
    S_fit = pd.concat(fits, axis=1)                                       # multi-column dataframe with gaussian fits. First level are the traps, second level are the rate windows for each trap
    Tm = S_fit.idxmax().unstack(level=0)                                  # Tm is the T at wich the fit has a maximum. This is a DataFrame with Rate windows as index and trap names as columns

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
        arrhenius_fits[trap] = pd.DataFrame(data = interc + slope*x, index = x)
        if not arrhenius_fits[trap].index.is_unique:
            warnings.warn('Some Temperature values of '+trap+"'s Arrhenius plot are duplicated, so I'm dropping the duplicated ones, keeping just the first.", stacklevel=2)
            arrhenius_fits[trap] = arrhenius_fits[trap][~arrhenius_fits[trap].index.duplicated(keep='first')]
    arrhenius_fit = pd.concat(arrhenius_fits, axis=1)
    arrhenius_fit.columns = arrhenius.columns.values

    return arrhenius, arrhenius_fit, S_fit, trap_params


###############################################################################################################



### PLOTTING + VISUALIZATION ###################################################

def plot_transients (tr, en_visualization = False, t1=None, t2 = None, t_4gates = None, cmap=None, **hvplot_opts):
    '''
    Plots the transients with an interactive widget allowing to visualize different temperatures. Returns an hvplot object.\n
    tr: Dataframe with time on index (default name 'Time (s)') and temperatures on columns (default name 'Temperature (K)')\n
    hvplot_opts: options to be passed to the hvplot() function. They can both overwrite the default options or add new ones\n
    en_visualization: shows a the transients with overlayed the position of t1 and t2 gates. If True, also t1 and t2 must be specified\n
    t1: t1 gates corresponding to the plotted spectrum (needed only if en_visualization==True)\n
    t2: t2 gates corresponding to the plotted spectrum (needed only if en_visualization==True)\n
    t_4gates: numpy array containing rate windows in rows and t1,t2,t3,t4 in columns (as returned by picts_4gates)
    cmap: colormap for the spectrum
    hvplot_opts: hvplot parameters to customize the spectrum plot.
    '''
    # Default options
    opts = dict(x='Time (s)', y=0, width=700, groupby = 'Temperature (K)',
                ylabel = 'Current (A)', color = 'k')
    # Overwrite or add the user specified options to the otpions used to produce the plot
    for opt in hvplot_opts:
        opts[opt] = hvplot_opts[opt]

    # Create plot
    plot = tr.stack().reset_index().hvplot(**opts)

    # Overlay rate window visualization if specified
    if en_visualization == True:
        picts_type = '2 gates'
        if (t1 is None and t2 is None and t_4gates is not None):
            picts_type = '4 gates'
        if (t_4gates is None and (t1 is None or t2 is None)): raise ValueError('You need to specify t1 and t2 or t4_gates if en_visualziation = True')

        # 2 gates visualization
        if (picts_type == '2 gates'):
            if cmap is None: colormap = hv.Cycle.default_cycles["default_colors"]
            else: colormap = cm(len(t1), cmap)
            lines = hv.Overlay([hv.VLine(x=t1[i]).opts(color=colormap[i])*\
                                hv.VLine(x=t2[i]).opts(color=colormap[i])\
                                for i in range(len(t1))])
        # 4 gates visualization
        else:
            if cmap is None: colormap = hv.Cycle.default_cycles["default_colors"]
            else: colormap = cm(len(t_4gates), cmap)
            lines = hv.Overlay([hv.VLine(x=t[0]).opts(color=colormap[i])*\
                                hv.VLine(x=t[1]).opts(color=colormap[i])*\
                                hv.VLine(x=t[2]).opts(color=colormap[i])*\
                                hv.VLine(x=t[3]).opts(color=colormap[i])
                                for i,t in enumerate(t_4gates)])

        return plot*lines

    else: return plot


