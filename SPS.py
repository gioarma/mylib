import re
import pandas as pd
from natsort import natsorted



def Import_SPS_Data (file_list, keyword = None, dropna=False, keep_keyword=False):

    '''
    Returns a Dataframe with multiindex columns. The upper level is [R, X, Y, Phase] and the lower is determined by the Keyword. The index of the DataFrame is the Energy.
    The function also searches for 1 file containing the 'Pyro' keyword for normalization.
    The values in Volts are converted to milliVolts for convenience.

    file_list: list of files to be imported. Each file corresponds to 1 spectrum, each spectrum to a DataFrame column\n
    name_kwd: keyword to searched in the file names in order to name the columns. name_kwd = 'Sample' and 'Sample1B' is contained in the file name -> column is named '1B' \n
    phase_analysis: whether to import also the X, Y and theta data in separate DataFrames, in addition to the R signal. In this case returns 4 DataFrames: R, X, Y, theta\n
    dropna: whether to drop all rows that contain NaN values. WARNING: use this if all you data contain NaNs in correspondence of the same energies (e.g. Pyro spectrum has 1nm spacing, while all samples have 2nm spacings)
    otherwise you will end up deleting some data (e.g. if one sample has 1nm spacing and the other 2nm spacing )\n
    keep_keyword: whether to keep the keyword in the column name. E.g. if name_kwd = 'Sample' and 'Sample1B' is contained in the file name -> column is named 'Sample1B' instead of '1B'

    '''

    if len(file_list) == 0:                                            # Check that data exist
        print ('WARNING: Data not found! Check the file path.')
        return None


# Define the column names and save them in 'col_names'
    col_names = []
    # If user specified a keyword
    if keyword != None:
        # This keep_keyword part is not working properly, to be fixed.
        if keep_keyword:
            pattern_1 = re.compile(r'[/_]('+keyword+r'[^_]+)[_. ]')
            pattern_2 = re.compile(r'[/_]([^_]+'+keyword+r')[_. ]')          # e.g. in '_20Hz_' with 'Hz' as keyword, this extracts '20Hz'.
        else:
            pattern_1 = re.compile(r'[/_]'+keyword+r'([^_]+)[_. ]')          # e.g. in 'aa/Sample1A_bb' with 'Sample' as keyword, this extracts '1A'. Same if you have '_' instead of '/'
            pattern_2 = re.compile(r'[/_]([^_]+)'+keyword+r'[_. ]')          # e.g. in '_20Hz_' with 'Hz' as keyword, this extracts '20Hz'.
        for file in file_list:                                       # Search for the keyword in file names in order to name the columns
            if 'Pyro' in file:                                       # We save the column of pyroelectric signal as 'Pyro'
                col_names.append('Lamp')
            else:
                match = re.findall(pattern_1, file)                                # Search the pattern in file
                if (len(match)==0): match = re.findall(pattern_2, file)            # Try with pattern 2 if pattern 1 does not match
                if (len(match)==0):
                    print('WARNING: No keyword matched for file'+file)             # Warning if pattern 2 does not work either
                elif (len(match)>1):
                    print('WARNING: More than one keyword matched for file'+file)  # Warning if more than 1 match
                col_names.append(match[0])
        # Code to try to convert all columns to integers. Handy beacuse you don't have to use '' for referring to columns, but it does not work properly with df.loc[] when you want to select a column range.
        #try:
        #    for i, col in enumerate(col_names):                                    # If possible convert to float the column names, useful in case of plots as a function of the column names
        #        if col != 'Lamp':                                                  # The 'Lamp' col is not convertible to a number
        #            if '.' in col: col_names[i] = float(col_names[i])              # Float number
        #            else: col_names[i] = int(col_names[i])                         # Int number
        #except: pass
    # If user has not specified a keyword:
    else: col_names=['Lamp' if 'Pyro' in f else 'S_'+str(i)
                     for i, f in enumerate(file_list)]

# Define the columns to be imported and import data
    cols = ['wavelength(nm)', 'R(V)', 'X(V)', 'Y(V)', 'theta(deg)']
    df = pd.concat([pd.read_csv(f, sep='\t', skiprows = 6, index_col = 'wavelength(nm)',
                                usecols = cols) for f in file_list], keys=col_names ,axis = 1)
# Adjustmets to DataFrame values
    df = df.swaplevel(axis=1)                                                    # Swap the multiindex along columns so that R, X, Y, theta are above
    df = df.sort_index(axis=1, level=0)                                          # Collapse all R, X, Y and theta columns together
    df.columns.set_levels(['R', 'X', 'Y', 'Phase'],level=0, inplace = True)      # Rename R(V) to R and so on
    df[['R', 'X', 'Y']] = df[['R', 'X', 'Y']].mul(1e3).values                    # Convert to mV from V
# Adjustments to DataFrame index
    df['Energy (eV)'] = 1239.8/df.index                                          # Create energy column
    df['Energy (eV)'] = df['Energy (eV)'].round(4)                               # Round to 4 decimal numbers to avoid index elements with too many decimals
    df = df.set_index('Energy (eV)')                                             # Set energy as index
    df = df.sort_index(axis = 0)                                                 # Sort energy index in ascending order
    unsorted_cols = df.columns.get_level_values(1).drop_duplicates()
    df = df.reindex(columns = df.columns.reindex(natsorted(unsorted_cols), level = 1)[0]) # Naturally sort columns, so that [100, 5, 70] becomes [5,70,100]
    if dropna: df = df.dropna()

    return df


############################



def Normalize_SPS_Data (df_NonNorm, norm_to_1 = False):
    '''
    Returns a copy of the passed DataFrame with R columns normalized by the lamp spectrum.
    Requires a 'Lamp' column in the passed DataFrame to normalize the other columns to.
    The X, Y and Phase columns are not normalized


    df_NonNorm: DataFrame to be normalized by the lamp spectrum
    norm_to_1: whether to normalize all columns to 1

    '''
    df = df_NonNorm.drop('Lamp', axis=1, level=1).copy()        # Copy whole non normalized df except 'Lamp' column
    df.columns = df.columns.remove_unused_levels()              # Have to do this otherwise the dropped column 'Lamp' is still listed in df.columns.levels
    df['R'] = df['R'].div(df_NonNorm['R']['Lamp'], axis = 0)\
              .mul(df.index, axis = 0)                          # Normalize by the lamp spectrum (S_norm = S_raw*E/Pyro)
    df['X']
    if norm_to_1: df/=df.max()                                  # Normalize to 1
    return df
