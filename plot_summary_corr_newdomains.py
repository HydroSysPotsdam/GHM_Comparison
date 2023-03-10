import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/correlations_partial.csv")
df_a = df.copy()

#df.loc[df["model_netrad"] == "no", "netrad"] = np.nan
df_a.loc[df_a["model_netrad"] == "yes", "netrad"] = np.nan
df_a.loc[df_a["model_netrad"] == "yes", "pr"] = np.nan

df_m = pd.melt(df, id_vars=['Test', 'output','GHM','Domain','model_netrad'], value_vars=['pr', 'netrad'], var_name='input', value_name='correlation')
df_a = pd.melt(df_a, id_vars=['Test', 'output','GHM','Domain','model_netrad'], value_vars=['pr', 'netrad'], var_name='input', value_name='correlation')

# todo: add data based correlations

d = df_m.loc[df_m["Test"] == "regression"] #spearman_partial
#d = df_m.loc[df_m["Test"] == "spearman"]
#d_p = df_m.loc[df_m["Test"] == "pearson"]

#  This could be accieved much easier
#t = d["correlation"].reset_index()["correlation"]
#t2 = d_p["correlation"].reset_index()["correlation"]
#d = d.reset_index()
#del d["index"]
#d["diff"] = t - t2

#d = d.loc[d["AI"] != "Whole model domain"]

#font = {'family' : 'sans-serif',
#        'sans-serif':'Helvetica',
#        'weight' : 'normal',
#        'size'   : 18}
#
#plt.rc('font', **font)


sns.set(font_scale=1.8)
sns.set_style("whitegrid", {'grid.linestyle': '--'})

#g = sns.catplot(y="GHM", x="correlation",
#                hue="Domain", col="input", row="output",
#                data=d, kind="strip", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], s=10, jitter=True, aspect=1.2);

#d.replace({"pr": "Precipitation [mm/yr]", "netrad": "Netradiation [mm/yr]", "evap": "Actual ET [mm/yr]", "qr": "Groundwater recharge [mm/yr]", "qtot": "Total runoff [mm/yr]"}, inplace=True)
d.replace({"pr": "Precipitation", "netrad": "Netradiation", "evap": "Actual ET", "qr": "Groundwater recharge", "qtot": "Total runoff"}, inplace=True)
#d.replace({'pr':'P', 'netrad':'N','evap':r'$E_a$', 'qr':'R', 'qtot':'Q'}, inplace = True)

d.rename(columns={"correlation":r"$\rho_s$"}, inplace=True)

models_r = {'clm45':'CLM4.5', 'jules-w1':'JULES-W1', 'lpjml':'LPJmL', 'matsiro':'MATSIRO', 'pcr-globwb':'PCR-GLOBWB', 'watergap2':'WATERGAP2', 'h08':'H08', 'cwatm':'CWatM'}
d.replace(models_r, regex=True, inplace=True)

d = d.sort_values('GHM')

g = sns.FacetGrid(col="input", row="output", data=d, aspect=1.5, height=4, margin_titles=True, gridspec_kws={"hspace":.7});


def plot_lin_diff(data, **kws):
    ax = plt.gca()
    d = data.loc[np.abs(data["diff"]) >= 0.1]
    d_a = d.loc[data["AI"] == 'Water-Limited']
    d_h = d.loc[data["AI"] == 'Energy-Limited']
    #d_s = d.loc[data["AI"] == 'Snow-Dominated']
    ax.scatter(d_h["correlation"] - d_h["diff"], d_h["GHM"], s=12, marker='D', color='#a1631d')
    ax.scatter(d_a["correlation"] - d_a["diff"], d_a["GHM"], s=12, marker='D', color='#318672')
    #ax.scatter(d_a["correlation"] - d_a["diff"], d_a["GHM"], s=12, marker='D', color='gray')

def plot_netrad(data, **kws):
    if data["input"].iloc[0] != "netrad":
        return
    ax = plt.gca()
    # get the name of the variable we are comparing against
    output = data["output"].iloc[0]
    date = df_a[df_a["output"] == output]
    date = date[date["input"] == "netrad"]
    date = date.loc[date["Test"] == "spearman"]
    sns.stripplot(x="correlation", y="GHM", hue="Domain", data=date, palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], size=10, marker="X")

def plot_scatter(data, **kws):
    ax = plt.gca()
    d1 = data[data["GHM"] == "Mean"]
    d2 = data[data["GHM"] != "Mean"]
    sns.scatterplot(data=d2, x=r"$\rho_s$", y="GHM", hue="Domain", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], s=80, ax = ax)
    sns.scatterplot(data=d1, x=r"$\rho_s$", y="GHM", hue="Domain", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], s=100, ax = ax, marker='^')
    #ax.plot(d1, x=r"$\rho_s$", y="GHM", hue="Domain", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], s=80, marker='^')

def plot_points(data, **kws):
    ax = plt.gca()
    d2 = data[data["GHM"] != "Mean"]
    #d2 = data.copy()
    #d2.loc[data["GHM"] == "Mean", r"$\rho_s$"] = np.nan
    #d2 = d2.replace("Mean", "Ensemble Mean")
    #d1 = data[data["GHM"] == "Mean"]
    sns.pointplot(data=d2, x=r"$\rho_s$", y="GHM", hue="Domain", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], scale=.5, dodge=False, ax=ax)
    #sns.pointplot(data=d1, x=r"$\rho_s$", y="GHM", hue="Domain", palette="BrBG_r", hue_order=["wet warm","wet cold","dry cold","dry warm"], scale=.001, dodge=False, ax=ax)

#g.map_dataframe(plot_lin_diff)
g.map_dataframe(plot_points)
g.map_dataframe(plot_scatter)
#g.map_dataframe(plot_netrad)
g.set(xlim=(-.25, 1))
#g.map(plt.axvline, x=0, linewidth=.5, linestyle="-", color='black', clip_on=False)
#sns.despine(offset=-150)
#g.set(xticks=[-1.,-0.75,-.5,-0.25,0,0.25,0.5,0.75,1], xticklabels=["-1","","-0.5","","0","","0.5","","1"])
g.set(xticks=[-0.25,0,0.25,0.5,0.75,1], xticklabels=["","0","","0.5","","1"])
#        yticklabels=["CLM4.5", "CWatM", "H08", "JULES-W1", "LPJmL", "MATSIRO", "PCR-GLOBWB", "WATERGAP2", "Mean"])
#g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_titles(col_template="", row_template="X = {row_name}")
g.tight_layout()

g.axes[2,0].set_xlabel(r'$\rho_s$(Precipitation, X)')
g.axes[2,1].set_xlabel(r'$\rho_s$(Net radiation, X)')

#g.axes[1,0].set_title('')
#g.axes[1,1].set_title('')

#g.axes[2,0].set_title('')
#g.axes[2,1].set_title('')


#g.savefig("summary_corr_normal.png", dpi=600)
#g.savefig("summary_corr_partial.png", dpi=600)
g.savefig("summary_regression.png", dpi=600)
