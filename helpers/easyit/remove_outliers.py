import numpy as np
import pandas as pd
# This function removes "outliers" based on a few criteria.

def remove_outliers_old(df):
    df.loc[df['qtot'] < 0, 'qtot'] = np.nan
    df.loc[np.logical_and(df['evap'] < 1, df['pr'] > 1000), 'qtot'] = np.nan

    df.loc[df['qr'] < 0, 'qr'] = 0
    df.loc[np.logical_and(df['evap'] < 1, df['pr'] > 1000), 'qr'] = np.nan

    #df['qs'].loc[df['qs']<0] = np.nan
    #df['qs'].loc[np.logical_and(df['evap']<1, df['pr']>1000)] = np.nan

    df.loc[df['evap'] <= 0, 'evap'] = np.nan
    df.loc[df['evap'] > 10000, 'evap'] = np.nan
    df.loc[np.logical_and(df['evap'] < 1, df['pr'] > 1000), 'evap'] = np.nan
    return df

# todo: remove same cells from all models
# todo: think about when to remove cells
# todo: get landmask for more objective removal of islands

def remove_outliers(df):

    df['qr'].loc[df['qr']<0] = 0 #  negative recharge values are set to 0
    df['evap'].loc[df['evap']>10000] = np.nan # extremely high evapotranspiration values are removed

    return df

def remove_outliers_old_seb(df):

    # use clm45 to remove islands and lakes
    path_clm45_evap = "2b/aggregated/clm45/evap.csv"
    df_clm45_evap = pd.read_csv(path_clm45_evap, sep=',')

    df['qtot'].loc[df['qtot']<0] = np.nan # negative runoff values are removed
    #df['qtot'].loc[np.logical_and(df['evap']<1, df['pr']>1000)] = np.nan # attempt to remove islands and lakes
    df['qtot'].loc[df_clm45_evap['evap']==0] = np.nan # attempt to remove islands and lakes

    df['qr'].loc[np.logical_and(df['qr']<0, df['qr']>-10)] = 0 # small negative recharge values (assumed capillary rise) are set to 0
    df['qr'].loc[df['qr']<-10] = np.nan # large negative recharge values are removed
    #df['qr'].loc[np.logical_and(df['evap']<1, df['pr']>1000)] = np.nan # attempt to remove islands and lakes
    df['qr'].loc[df_clm45_evap['evap']==0] = np.nan # attempt to remove islands and lakes

    df['qs'].loc[df['qs']<0] = np.nan # negative surface runoff values are removed
    #df['qs'].loc[np.logical_and(df['evap']<1, df['pr']>1000)] = np.nan # attempt to remove islands and lakes
    df['qs'].loc[df_clm45_evap['evap']==0] = np.nan # attempt to remove islands and lakes

    df['evap'].loc[df['evap']<=0] = np.nan # negative evapotranspiration values are removed
    df['evap'].loc[df['evap']>10000] = np.nan # extremely high evapotranspiration values are removed
    #df['evap'].loc[np.logical_and(df['evap']<1, df['pr']>1000)] = np.nan # attempt to remove islands and lakes
    df['evap'].loc[df_clm45_evap['evap']==0] = np.nan # attempt to remove islands and lakes

    return df
