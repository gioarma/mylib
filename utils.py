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
