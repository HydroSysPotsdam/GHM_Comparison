from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce
from helpers.easyit.easyit import EasyIt
from helpers.easyit.easyit import load_data
from scipy import stats
import numpy.polynomial.polynomial as poly
from pingouin import partial_corr

# prepare data
#data_path = "../data/"
data_path = "model_outputs/2b/aggregated/"

ghms = ["clm45", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2", "h08", "cwatm"] #["watergap2-2c_nosoc"]#,
# "watergap2-2c_varsoc"]#

#outputs = ["evap", "qr", "qs", "qsb", "qtot"]
outputs = ["evap", "qr", "qtot"]
#outputs = ["evap", "qr"]
# additional_outputs = ["swe"]

#forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "netrad"]
forcings = ["pr", "netrad_median"]


from scipy.stats import t as ttest

def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


def applyTest(X, y, lvar, test, binning=False):
    results = {}
    for v in lvar:
        _x = X[v]
        _y = y

        assert len(_x) == len(y)
        if binning:
            s = stats.binned_statistic(_x, _y, 'mean', bins=100)
            _y = s.statistic
            bin_edges = s.bin_edges
            bin_width = (bin_edges[1] - bin_edges[0])
            _x = bin_edges[1:] - bin_width/2

        if test == "pearson":
            k, p1 = stats.normaltest(_x)
            k, p2 = stats.normaltest(_y)
            #alpha = 1e-3
            #print(v)
            #if p1 < alpha:
            #    print("X no normal dis")
            #else:
            #    print("X normal dis.")
            #if p2 < alpha:
            #    print("y no normal dis")
            #else:
            #    print("y normal dis.")
            r, p = stats.pearsonr(_x,_y)
            results[v] = r
        if test == "spearman":
            r, p = stats.spearmanr(_x,_y, nan_policy='omit')
            results[v] = r
        if test == "spearman_partial":
            df_tmp = pd.DataFrame()
            if v=='pr':
                _z = X['netrad']
            elif v=='netrad':
                _z = X['pr']
            df_tmp["x"] = _x
            df_tmp["y"] = _y
            df_tmp["z"] = _z
            r_partial_mat = partial_corr(data=df_tmp, x="x", y="y", covar="z", method='spearman')
            #r_partial_mat = partial_corr(data=df_tmp, x="x", y="y", x_covar="z", method='spearman')
            results[v] = r_partial_mat["r"].values[0]

        if test == "regression":
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(_x.array.reshape(-1, 1), _y.array)
            b1 = reg.coef_
            results[v] = b1[0]

        if test == "lin-reg_r-seq":
            r = stats.linregress(_x, _y)
            results[v] = r.rvalue**2
        if test == "lin-reg_slope_err":
            r = stats.linregress(_x, _y)
            tinv = lambda p, df: abs(ttest.ppf(p/2, df))
            ts = tinv(0.05, len(_x)-2)
            results[v] = ts*r.stderr
        if test == "lin-reg_intercept_err":
            r = stats.linregress(_x, _y)
            tinv = lambda p, df: abs(ttest.ppf(p/2, df))
            ts = tinv(0.05, len(_x)-2)
            results[v] = ts*r.intercept_stderr
        if test == "lin-reg_slope":
            r = stats.linregress(_x, _y)
            if r.rvalue**2 < 0.7:
                results[v] = np.nan
            else:
                results[v] = r.slope
        if test == "lin-reg_intercept":
            r = stats.linregress(_x, _y)
            results[v] = r.intercept
        if test == "weighted_lin-reg_slope":
            #https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix
            from sklearn.linear_model import LinearRegression
            # fit WLS using sample_weights
            WLS = LinearRegression()
            # give a 0 weight to all outliers; = > mean + SD
            w = np.ones(len(_x))
            cond = _x.mean() + np.std(_x)
            cond_i = np.where(_x >= cond)[0]
            for i in cond_i:
                w[i] = 0.0
            X_i = _x.to_numpy().reshape(-1, 1)
            WLS.fit(X_i, _y, sample_weight=w)
            points = np.linspace(_x.min(), _x.max(), len(_x))
            rsq = WLS.score(X_i, _y, sample_weight=w)
            slope = WLS.coef_[0]

            if rsq < 0.7:
                results[v] = np.nan
            else:
                results[v] = slope
        if test == "pol._rsq":
            x = X[v]
            coefs= poly.polyfit(x.to_numpy(),_y.to_numpy(),2)
            points = np.linspace(x.min(), x.max(), len(x))
            ffit = poly.polyval(points, coefs)
            rsq = r_squared(_y, ffit)
            results[v] = rsq
        if test == "lin._corr-ratio":
            bin_edges = stats.mstats.mquantiles(_x, np.linspace(0,1,11))
            nx = bin_edges.size - 1
            #mean_stat = stats.binned_statistic(_x, _y, statistic=lambda y: np.nanmean(_x), bins=bin_edges)
            #c1 = nx * np.power((mean_stat.statistic - np.nanmean(_y)),2)
            #c2 = nx * np.power((_y.to_numpy() - np.nanmean(_y)),2)
            #corr_ratio = c1.sum() / c2.sum()
            #results[v] = corr_ratio
            results[v] = 0
        if test == "pol._corr-ratio":
            bin_edges = stats.mstats.mquantiles(_x, np.linspace(0,1,11))
            nx = bin_edges.size - 1
            coefs = poly.polyfit(_x.to_numpy(),_y.to_numpy(),2)
            ffit = poly.polyval(np.linspace(0.001, 10000, 10000), coefs)
            ffit_e = poly.polyval(_x.to_numpy(), coefs)
            c1 = nx * np.power((ffit_e - np.nanmean(_y)),2)
            c2 = nx * np.power((_y.to_numpy() - np.nanmean(_y)),2)
            corr_ratio_pol = c1.sum() / c2.sum()
            results[v] = corr_ratio_pol

    return results

#tests = ["pearson"]
#tests = ["spearman", "lin-reg_r-seq"]
#tests = ["spearman", "lin._corr-ratio", "pol._corr-ratio"]

#tests= ["spearman_partial","spearman","regression"]
tests= ["spearman"]

#cats = ['Energy-Limited', 'Water-Limited']

#from lib.remove_outliers import remove_outliers

df_f = load_data(data_path, forcings)

#df_domains = pd.read_csv("../data/domains.csv")
df_domains = pd.read_csv("model_outputs/2b/aggregated/domains.csv")
a = "domain_days_below_1_0.08_aridity_netrad"
#a = "domain_days_below_1_0.08_aridity_potevap"
#a = "domain_days_below_6.7_0.25_aridity_netrad"
#a = "domain_days_below_6.7_0.25_aridity_potevap"

df_domains = df_domains[["lat","lon",a]]
df_domains.rename(columns={a: "Domain"}, inplace=True)

l = []
l_allmodels = []
for df, g in EasyIt(facs=outputs, data_path=data_path):
    d = pd.merge(df, df_f, on=['lat', 'lon'])
    if g in ["h08", "cwatm", "jules-w1", "watergap2", "matsiro", "clm45"]:
        del d["netrad"]
        data = pd.read_csv(Path(data_path).joinpath(g, "netrad.csv"))
        d = pd.merge(d, data, on=['lat', 'lon'])
     
    #d["netrad"] = d["netrad"] / 2257 * 0.001 * (60* 60 *24 *365)
    d["netrad"] = d["netrad"] *12.87
    #d = remove_outliers(d)
    #d = d.dropna()
    d = d[d["evap"] < 10000]
    d.loc[d['qr'] < 0, 'qr'] = 0
   

    #lat    lon        evap         qr        qtot         pr      netrad    Domain
    d["model"] = g
    l_allmodels.append(d)

    d = pd.merge(d, df_domains, on=['lat', 'lon'])

    f = ["pr", "netrad"]
    #categories = ['Energy-Limited', 'Water-Limited', 'Snow-Dominated']
    categories = ['wet warm', 'wet cold', 'dry cold', 'dry warm']
    for c in categories:
        df_c = d.loc[d["Domain"] == c]
        for o in outputs:
            for t in tests:
                data = applyTest(df_c[f], df_c[o], f, t)
                data['Test'] = t
                data['output'] = o
                data['GHM'] = g
                data['Domain'] = c
                if g in ["h08","cwatm","jules-w1","watergap2","matsiro", "clm45"]:
                    data['model_netrad'] = "yes"
                else:
                    data['model_netrad'] = "no"
                l.append(data)
        #print("Sis: {}, GHM: {}, output: {}".format(si['CV'],g,i))
#    for o in outputs:
#        for t in tests:
#            data = applyTest(d[f], d[o], f, t)
#            data['Test'] = t
#            data['output'] = o
#            data['GHM'] = g
#            data['Domain'] = "Whole model domain"
#            if g in ["h08","cwatm","jules-w1","watergap2"]:
#                data['model_netrad'] = "yes"
#            else:
#                data['model_netrad'] = "no"
#            l.append(data)


df_all = pd.concat(l_allmodels)
df_mean = df_all.groupby(["lat", "lon"]).mean(numeric_only=True).reset_index()
df_mean = pd.merge(df_mean, df_domains, on=['lat', 'lon'])

f = ["pr", "netrad"]
#categories = ['Energy-Limited', 'Water-Limited', 'Snow-Dominated']
categories = ['wet warm', 'wet cold', 'dry cold', 'dry warm']
for c in categories:
    df_c = df_mean.loc[df_mean["Domain"] == c]
    for o in outputs:
        for t in tests:
            data = applyTest(df_c[f], df_c[o], f, t)
            data['Test'] = t
            data['output'] = o
            data['GHM'] = "Mean"
            data['Domain'] = c
            data['model_netrad'] = "yes"
            l.append(data)

df = pd.DataFrame(l)
#df.to_csv("results/correlations_partial.csv", index=False)
df.to_csv("results/correlations_new_domains_new.csv", index=False)
