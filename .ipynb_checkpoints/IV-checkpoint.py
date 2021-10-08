import pandas as pd
import re

def Read_IV_Data_Probestation (data, keyword = None, zero_start = False):

    '''
    data: list of filepths to the data files
    keyword: to search in the datafile for column naming
    zero_start: whether the measurement was taken with Zero start option

    The forward and backward scans are returned as two distinct dfs
    '''

    if len(data) == 0:
        print ('WARNING: Data not found! Check the file path.')
        return None
    dfs_forw = {}  # Dictionary with all the forward IVs
    dfs_back = {}  # Dictionary with all the backward IVs
    df_names = []

    # If user specified a keyword, then we get the column names from the data files
    if keyword != None:
        pattern_1 = re.compile(r'[/_]'+keyword+r'([^_]+)[_.]')         # e.g. in 'aa/Sample1A_bb' with 'Sample' as keyword, this extracts '1A'. Same if you have '_' instead of '/'
        pattern_2 = re.compile(r'[/_]([^_]+'+keyword+r')[_.]')         # e.g. in '_20Hz_' with 'Hz' as keyword, this extracts '20Hz'.
        for file in data:
             # Search for the keyword in file name in order to name the dataframes
            match = re.findall(pattern_1, file)                      #Search the pattern in file
            if (len(match)==0): match = re.findall(pattern_2, file)  # Try with pattern 2 if pattern 1 does not match
            if (len(match)==0): print('WARNING: No keyword matched for file'+file)             # Warning if pattern 2 does not work either
            elif (len(match)>1): print('WARNING: More than one keyword matched for file'+file) # Warning if more than 1 match
            df_names.append(match[0])
    # If user didn't specify a keyword use standard names for currents: I_1, I_2, etc.
    else:
        for i in range(len(data)):
            df_names.append('I_'+str(i))

    # Import the data
    for i, file in enumerate(data):
        df = pd.read_csv(file, sep = '\t', header = 1,
                        usecols = ['#VGate(V)', 'IGate(A)'])
        df.columns = ['V', df_names[i]]
        df.set_index('V', inplace = True)
        # Calculate the absolute value of the current
        abs_name = 'abs_'+df_names[i]
        df[abs_name] = abs(df.iloc[:, 0])

        #Separate into forward and backward scans
        if zero_start == False:
            N = df.index.size # Num of points in the total IV
            df_forw = df.iloc[:N//2,:].copy()
            df_back = df.iloc[N//2:,:].copy()
        else:
            df.drop(df.tail(1).index, inplace = True) # Delete last row, which contains a duplicate 0V measurement (we already have 2, one at the beginning and one in the middle)
            N = df.index.size # Num of points in the total IV
            df_forw = df.iloc[N//4:3*N//4,:].copy()
            df_back = pd.concat([df.iloc[:N//4+1,:].copy(),
                                 df.iloc[3*N//4:N,:].copy()])
            df_back.sort_index(ascending = False, inplace = True)  #Order from positive to negative voltages

        # Add the df to the dictionary of dfs
        dfs_forw.update({df_names[i]: df_forw})
        dfs_back.update({df_names[i]: df_back})

    # Put all dfs in a single df
    df_Tot_forw = pd.concat([dfs_forw[i] for i in dfs_forw], axis = 1)
    df_Tot_back = pd.concat([dfs_back[i] for i in dfs_back], axis = 1)

    return df_Tot_forw, df_Tot_back



def Read_IV_Data_IV2012LabView (path):
    df = pd.read_csv(path, sep='\t', index_col='V')
    df.columns = [path]
    return df
