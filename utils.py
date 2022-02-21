from lmfit.models import LinearModel
import pandas as pd

def filter_df (df, filter_dict):
    '''
    Filters a DataFrame keeping only the rows that contain the values indicated in the filter
    df: DataFrame to be filtered. \n
    filter_dict: dictionary where the keys are column names of the DataFrame and the values are lists of values to be searched in the columns. Only the rows that contain these values are kept. \n\n
    Returns: pd.DataFrame 
    '''
    df = df.copy()
    for key, items in filter_dict.items():
        df = df[df[key].isin(items)]
    return df


def fit_df (df, model, params=True):
    '''
    Returns a copy of the input dataframe where all columns contain the best fit to the desired model of the corresponding column using lmfit library.
    df: DataFrame where each column has to be fitted with the desired model and the x-data are in the index. If IV contains any NaNs, they will be dropped.
    model: fitting model of the lmfit library to be used. Currently implemented: ['linear'].
    params: if True the function also returns a dict of lmfit.parameter object with the output parameters of the fit. The keys of the dictionary correspond to the column names
    '''
    
    if model=='linear': mod = LinearModel()
    else: raise NotImplementedError("This model is not yet implemented.")
    
    if isinstance(df, pd.Series): df = df.to_frame()
    
    fits = {}
    params = {}
    for col in df:
        y = df[col].dropna()
        x = y.index
        out = mod.fit(data=y, x = x)
        fits[col] = pd.DataFrame(data=out.best_fit.values, index=x)
        params[col] = out.params
    df_fit = pd.concat(fits, axis=1)
    df_fit.columns = df.columns           # With multiindex it can happen that a new unwanted level with all 0 values is added, this fixes it (there's probably a more clever way)
    if params: return df_fit, params
    else: return df_fit