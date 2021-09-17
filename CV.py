import pandas as pd, numpy as np, holoviews as hv
import re
from scipy.signal import savgol_filter
from scipy.constants import epsilon_0, e

def Import_CV_Data (file_list, keyword = None, f_sweep = False):

    '''
    Imports data of Capacitance Voltage measurements done with LCZ meter.

    file_list: list of paths to the data files
    keyword: look for this keyword in the data file names to give columns
    a certain name. E.g. if the file is `.../Sample1A_foo.txt` and keyword is
    `Sample`, then the data are imported with column name `1A`
    f_sweep: If True, for each file the measurement frequency is read and a
    multiindex Series is returned, with frequency as external index and bias
    as internal index. For the moment this assumes that each file corresponds
    to a different frequency. It still does not support the case where you have,
    e.g., different frequency sweeps on different samples.
    '''

    # Import data
    df = pd.concat([pd.read_csv(f, sep = '\t', header = 5, index_col = 'Bias (V)')
                    for f in file_list], axis = 1)

    # Determine column names
    # Case of frequency sweeps
    if f_sweep:
        df.columns = [float(pd.read_csv(f, sep='\t').iloc[1,1]) for f in file_list]    # Read probe frequency from data file, which is in position [1,1] for LCZ meter
        df = df.unstack()                                                              # Convert Dataframe into multiindex series
        df.sort_index(inplace = True)                                                  # Reorder the indices so that the frequency values are in increasing order
        df.index.rename(['Frequency (Hz)','Bias (V)'], inplace = True)                 # Rename indices
    # No frequency sweep
    else:
        if keyword!=None:
            col_names = []
            pattern_1 = re.compile(r'[/_]'+keyword+r'([^_]+)[_.]')         # e.g. in 'aa/Sample1A_bb' with 'Sample' as keyword,
                                                                           # this extracts '1A'. Same if you have '_' instead of '/'
            pattern_2 = re.compile(r'[/_]([^_]+'+keyword+r')[_.]')         # e.g. in '_20Hz_' with 'Hz' as keyword, this extracts '20Hz'.
                                                                           # In this case we extract also the keyword itself
            for file in file_list:
                match = re.findall(pattern_1, file)                                                # Search the pattern in file
                if (len(match)==0): match = re.findall(pattern_2, file)                            # Try with pattern 2 if pattern 1 does not match
                if (len(match)==0): print('WARNING: No keyword matched for file'+file)             # Warning if pattern 2 does not work either
                elif (len(match)>1): print('WARNING: More than one keyword matched for file'+file) # Warning if more than 1 match
                col_names.append(match[0])

        else: col_names=['C_'+str(i) for i in range(len(file_list))]

        df.columns = col_names
        df.sort_index(axis = 1, inplace = True)   # Sort columns in alphabetical order

    return df

################################

def Del_Duplicates(df):
    indices = [list(df[df.diff()[i]==0].index) for i in df.columns] # index of a df containing only rows where the value of i-th column is equal to the previous one (diff()==0)
    for col, rows in enumerate(indices):
        for row in rows:
            df.loc[row, df.columns[col]] = np.nan
    return df


################################


def Smooth (df, window_length, polyorder):

    '''
    Smoothes each column with Savitsky-Golay filtering

    df: input dataframe with
    window_length: window length of SG filter. Has to be higher than polyorder
    polyorder: polynomial order of SG filter, which needs to be an odd integer.
    '''

    return pd.DataFrame(savgol_filter(df, window_length,
                                      polyorder, axis = 0),
                        columns = df.columns,
                        index = df.index)



################################


def Diff (df):
    '''
    Takes the 1st derivative of each column of the input DataFrame with second order central differences method (numpy)

    CV: input DataFrame
    '''
    col_list = []             # I have to do this calculation for each column to drop NaNs that could be in a given col. If I do this on the whole df using dropna() I would loose some data because the method drops the whole row containing some NaN values
    for i in df.columns:
        col_list.append(pd.Series(np.gradient(df[i], df[i].index),
                                  index = df[i].index.values, name = i))

    return pd.concat(col_list, axis = 1)



##############################



def Calculate_CarrierProfile (CV, epsilon, A, smooth = False, window=None, polyorder=None, warnings = True):
    '''
    Returns a Dataframe with the carrier concentration in cm-3 as columns and profiling depth in µm as index.

    CV: Dataframe with CV data as a function of V
    epsilon: Dielectric constant of the semiconductor
    A: Contact area in cm^2
    smooth: Whether the KV curve should be smoothed with Savitsky-Golay filter before differentiating
    window: Window length of SG filter. Has to be odd
    polyorder: Polynomial order of SG filter, which needs to be higher than window.
    warnings: Whether to print warnings about processes that went wrong during execution
    '''

    KV = 1/CV**2

    if smooth:
        if window==None or polyorder==None:
            print('WARNING: window and polyorder must be set if smoothing is on.')
            return 0
        KV = Smooth(KV, window, polyorder)

    dKV = Diff(KV)
    dKV.replace(0, np.nan, inplace = True)       # Remove 0 values that lead to divergencies

    A_m2 = A*1e-4       # Convert area in m^2

    # Dict of series, with N as values and w as index:
    Nw = {i: pd.Series(-2/(e*epsilon*epsilon_0*A_m2**2*dKV[i].values)*1e-6,
                        index = epsilon*epsilon_0*A_m2/CV[i].values*1e6, name = i)
          for i in CV.columns}

    for N in Nw.values():
        N.dropna(inplace = True)          # Drop NaNs in each series values
        N = N.loc[N.index.dropna()]       # Drop NaNs in each index

    #return Nw
    try:
        return pd.concat(Nw.values(), axis = 1)
    except ValueError:
        if warnings:
            print('''WARNING: There are some depth values associated to more than one concentration value. To avoid this, smooth the CV curves using `smooth=True`.\n
Due to this, a dictionary containing all the curves has been returned, instead of a single DataFrame.
You can access curve `x` with `Nw['x']`.''')
        for i in Nw.values(): i.sort_index(inplace=True)     # Reorder the index of each series. This is an operation that pd.concat() also does.
        return Nw


###############################

def Plot_Profiling(Nw, N_min = 0, N_max = 1e24, **opts):

    '''
    Returns the plot of the carrier profiling, both if the data are saved in a dict
    (when the concat operation has failed because of more than one N for a given w),
    or in a DataFrame.
    It removes negative concentration values to avoid problems in log plots

    Nw: Profile to be plotted
    N_min: Only N values above this are plotted
    N_max: Only N values below this are plotted
    **opts: options to send to hvplot
    '''

    if type(Nw)==dict:
        return hv.Overlay([i[(i>N_min) & (i<N_max)].plot(**opts) for i in Nw.values()])
    else:
        return Nw[(Nw>N_min) & (Nw<N_max)].plot(group_label=' ', **opts)
