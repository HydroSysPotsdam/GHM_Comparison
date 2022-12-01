from functools import reduce
import pandas as pd
from helpers.easyit.remove_outliers import remove_outliers

def load_data(path, f_list):
    l = []
    for f in f_list:
        df = pd.read_csv(path + f + ".csv")
        l.append(df)
    return reduce(lambda x, y: pd.merge(x, y, on=['lat', 'lon']), l)


class EasyIt(object):
    def __init__(self, ghms=None, facs=None, data_path="data/", loop="ghms"):
        self.ghms = ["clm45", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2", "h08", "cwatm"] if ghms is None else ghms
        self.facs = ["dis", "evap", "qr", "qs", "qsb", "qtot"] if facs is None else facs
        self.path = data_path
        self.loop = loop
        self.data = None
        self.i = None

    def __iter__(self):
        if self.loop == "ghms":
        # TODO support iteration over factors and then ghms as well
            for i in self.ghms:
                self.data = load_data(self.path + i + "/", self.facs)
                self.i = i
                yield self.data, self.i


# loads forcings and outputs for all models in a format that can be used by seaborn
def load_data_all(path, f_list, o_list=None, g_list=None, rmv_outliers=True):
    df_f = load_data(path, f_list)
    df = pd.DataFrame()
    for df_tmp, g in EasyIt(ghms=g_list, facs=o_list, data_path=path):
        if rmv_outliers:
            df_tmp = remove_outliers(df_tmp) # only works if all variables are parsed
        df_tmp = pd.merge(df_f, df_tmp, on=['lat', 'lon'])
        df_tmp["ghm"] = g
        df = pd.concat([df, df_tmp])
    df = df.dropna()
    return df
