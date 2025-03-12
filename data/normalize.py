

def min_max_normalize(col):
    """ Normalize column using Min-Max normalization. (0, 1)"""
    return (col - col.min()) / (col.max() - col.min())

def min_max_normalize_mod(col):
    """ Normalize column using Min-Max normalization. (-1, 1) """
    return 2 * ((col - col.min()) / (col.max() - col.min())) - 1

def standarization(col):
    """ Standarize column using Z-score normalization. """
    return( (col - col.mean()) / col.std() )

def normalize_df(df, method="min-max"):
    """ Normalize the dataframe and return it. """
    for col in df.columns:
        if method == "min-max":
            #* Min Max normalization makes data uniform, small ranged & safer for neural networks
            df[col] = min_max_normalize(df[col])
        if method == "min-max-mod":
            #* Improved min-max into range (-1, 1)
            df[col] = min_max_normalize_mod(df[col])
        elif method == "z-score":
            df[col] = standarization(df[col])
    return df
