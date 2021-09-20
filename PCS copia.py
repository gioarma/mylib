import pandas as pd
import numpy as np
import re



def Import_PCS_Data(file_list, keyword = None):

    '''
    file_list: list of files to be imported. Each file corresponds to 1 spectrum, each spectrum to a DataFrame column
    keyword: keyword to be searched in the file names in order to name the columns. E.g. keyword = 'Sample' and 'Sample1B' is contained in the file name -> column is named '1B'

    Creates automatically the energy column from the wavelength data and sets it as df index.
    Also searches for 1 file containing the 'Pyro' keyword for normalization.
    '''

    # Read the data and arrange them in a Dataframe
    df = pd.concat([pd.read_csv(f, sep = '\t', decimal=',', skiprows=10,
                                usecols=[0,1], names = ['Wavelength', 'R'],
                                index_col='Wavelength')
                    for f in file_list], axis = 1)
    df.drop(df.index[0], inplace = True)      # I can't ignore this row from the read_csv method (using skiprows) without getting an incorrect import (bug?)
    df = df.astype(float)                     # The imported columns seem to contain strings so I convert them to float to be able to make calculations on them (bug?)
    df.index = df.index.astype(float)
    # Define the column names and save them in 'col_names'
    col_names = []
    if keyword != None:
        pattern_1 = re.compile(r'[/_]'+keyword+r'([^_]+)[_.]')         # e.g. in 'aa/Sample1A_bb' with 'Sample' as keyword,
                                                                 # this extracts '1A'. Same if you have '_' instead of '/'
        pattern_2 = re.compile(r'[/_]([^_]+'+keyword+r')[_.]')         # e.g. in '_20Hz_' with 'Hz' as keyword, this extracts '20Hz'.
                                                                 # In this case we extract also the keyword itself
        for file in file_list:
            if 'Pyro' in file:  col_names.append('Pyro')                                 # We save the column of pyroelectric signal as 'Pyro'

            else:
                match = re.findall(pattern_1, file)                                  # Search the pattern in file
                if (len(match)==0): match = re.findall(pattern_2, file)              # Try with pattern 2 if pattern 1 does not match

                if (len(match)==0): print('WARNING: No keyword matched for file'+file)             # Warning if pattern 2 does not work either
                elif (len(match)>1): print('WARNING: More than one keyword matched for file'+file) # Warning if more than 1 match

                col_names.append(match[0])

    else: col_names=['S_Pyro' if 'Pyro' in f else 'S_'+str(i)
                     for i, f in enumerate(file_list)]

    df.columns = col_names
    df.sort_index(axis = 1, inplace = True)   # Sort columns in alphabetical order

    #Create the energy column and set it as index
    df['Energy (eV)'] = 1239.8/df.index.values
    df.set_index(['Energy (eV)'], inplace=True)
    return df


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def Normalize_PCS_Data (df_NonNorm, norm_to_1 = False):
    '''
    df_NonNorm: DataFrame to be normalized by the lamp spectrum
    norm_to_1: whether to normalize all columns to 1

    Requires a 'Pyro' column in the passed DataFrame
    '''
    df = df_NonNorm.drop('Pyro', axis = 1).copy()    #Copy whole non normalized df except 'Pyro' and 'Wavelength' columns
    df = df.div(df_NonNorm['Pyro'], axis = 0)\
           .mul(df.index, axis = 0)                  # Normalize by the lamp spectrum (S_norm = S_raw*E/Pyro)
    if norm_to_1: df/=df.max()                       # Normalize to 1
    return df



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def Make_Tauc_Plots (df):

    '''
    df: DataFrame whose columns have to be turned into Tauc plots
    For the moment only for the case of Direct gap.
    '''
    df_fit = df.copy()
    for i in df_fit:
         df_fit[i] = (df[i]*df.index)**(2)     # For Direct Gap -> (alpha*E)^2 vs E
    return df_fit
