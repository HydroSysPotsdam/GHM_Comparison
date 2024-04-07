import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as ml_colors
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from brewer2mpl import brewer2mpl
import os
from functools import reduce
import random
import matplotlib.colors
from pingouin import partial_corr
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

# This script contains a collection of function that make or help with making plots.

def plot_origin_line(x, y, **kwargs):

    ax = plt.gca()
    lower_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='grey', alpha=0.5, zorder=1)


def plot_origin_line_alt(ax=plt.gca(), **kwargs): #x, y,

    #ax = plt.gca()
    lower_lim = [ax.get_ylim()[0]]
    upper_lim = [ax.get_ylim()[1]]
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='grey', alpha=0.5, zorder=1)


def plot_Budyko_limits(x, y, **kwargs):

    ax = plt.gca()
    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', c='gray')
    ax.plot(np.linspace(1, lim, 100), np.linspace(1, 1, 100), '--', c='gray')


def plot_coloured_scatter_random_domains(x, y, domains, alpha=0.1, s=1, **kwargs):

    # extract data
    df = kwargs.get('data')
    df_shuffled = df.sample(frac=1)

    # specify colours
    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}
    df_shuffled["colour"] = df_shuffled[domains]
    df_shuffled = df_shuffled.replace({"colour": palette})
    df_shuffled = df_shuffled.dropna(subset = ['colour'])

    # make plot
    ax = plt.gca()
    plt.scatter(df_shuffled[x], df_shuffled[y], marker='o', alpha=alpha, s=s, lw = 0, c=df_shuffled["colour"])


def plot_coloured_scatter_random(x, y, domains, palette, alpha=0.1, s=1, **kwargs):

    # domains and palette have to fit
    #palette = {"energy-limited": 'tab:blue', "water-limited": 'tab:orange'}

    # extract data
    df = kwargs.get('data')
    df_shuffled = df.sample(frac=1)
    df_shuffled = df_shuffled.dropna()

    # specify colours
    df_shuffled["colour"] = df_shuffled[domains]
    df_shuffled = df_shuffled.replace({"colour": palette})

    # make plot
    ax = plt.gca()
    plt.scatter(df_shuffled[x], df_shuffled[y], marker='o', alpha=alpha, s=s, lw=0, c=df_shuffled["colour"])


def plot_bins_group(x, y, color="tab:blue", group_type="aridity_class", group="energy-limited", **kwargs):

    # extract data
    df = kwargs.get('data')

    # get correlations
    df = df.dropna()
    df_group = df.loc[df[group_type]==group]

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(df_group[x], df_group[y])

    ax = plt.gca()
    corr_str = ''
    r_sp, _ = stats.spearmanr(df.loc[df[group_type] == group, x], df.loc[df[group_type] == group, y], nan_policy='omit')
    #corr_str = corr_str + r' $\rho_s$ ' + str(group) + " = " + str(np.round(r_sp,2))
    corr_str = corr_str + str(np.round(r_sp,2))
    print(corr_str)
    r_sp_tot, _ = stats.spearmanr(df[x], df[y], nan_policy='omit')
    print("(" + str(np.round(r_sp_tot,2)) + ")")

    # plot bins
    ax = plt.gca()
    ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
                fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)


def plot_lines(x, y, ax, palette, domains, domain, n=11, ls="solid"):

    import matplotlib.patheffects as mpe
    outline = mpe.withStroke(linewidth=4, foreground='white')

    color = palette[domain]

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(x, y, n)

    r_sp, _ = stats.spearmanr(x, y, nan_policy='omit')
    if r_sp < 0.1:
        r_sp = 0.1

    #ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
    #            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)
    ax.plot(bin_median, median_stat.statistic, color=color, alpha=0.8, linestyle=ls) #, path_effects=[outline]
    #ax.fill_between(bin_median, p_25_stat.statistic, p_75_stat.statistic, facecolor=color, alpha=0.1)


def plot_lines_group(x, y, palette, domains, domain, n=11, **kwargs):

    import matplotlib.patheffects as mpe
    outline = mpe.withStroke(linewidth=4, foreground='white')

    # extract data
    df = kwargs.get('data')

    # get correlations
    #df = df.dropna()
    df_group = df.loc[df[domains]==domain]
    color = palette[domain]

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(df_group[x], df_group[y], n)

    ax = plt.gca()
    corr_str = ''
    r_sp, _ = stats.spearmanr(df.loc[df[domains] == domain, x], df.loc[df[domains] == domain, y], nan_policy='omit')
    #corr_str = corr_str + r' $\rho_s$ ' + str(domain) + " = " + str(np.round(r_sp,2))
    corr_str = corr_str + str(np.round(r_sp,2))
    print(corr_str)
    r_sp_tot, _ = stats.spearmanr(df[x], df[y], nan_policy='omit')
    print("(" + str(np.round(r_sp_tot,2)) + ")")

    # plot bins
    ax = plt.gca()
    #ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
    #            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)
    ax.plot(bin_median, median_stat.statistic, color=color, path_effects=[outline])
    #ax.fill_between(bin_median, p_25_stat.statistic, p_75_stat.statistic, facecolor=color, alpha=0.1)
    #ax.fill_between(bin_median, p_05_stat.statistic, p_95_stat.statistic, facecolor=color, alpha=0.1)


def plot_bins(x, y, n=11, **kwargs):

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(x, y, n)

    # plot bins
    ax = plt.gca()
    color = 'black'
    ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
                fmt='o', ms=4, elinewidth=1, c=color, ecolor=color, mec=color, mfc='white', alpha=0.9)
    ax.plot(bin_median, median_stat.statistic, c=color, alpha=0.5, linewidth=1.0)

    lim=ax.get_ylim()[1]
    ax.plot(np.linspace(np.min(x), np.max(x), 1000), 0.9 * lim * np.ones(1000),
            '-', c='black', alpha=0.2, linewidth=8, solid_capstyle='butt')
    ax.errorbar(bin_edges[1:], 0.9*lim*np.ones(10), xerr=None, yerr=0.025*lim*np.ones(10),
                c='white', zorder=10, fmt='none')


def binned_stats_table(df, x_str, y_str, ghms, n=11):

    l = []
    for g in ghms:
        x = df.loc[df["ghm"]==g, x_str]
        y = df.loc[df["ghm"]==g, y_str]

        # calculate binned statistics
        bin_edges, \
        mean_stat, std_stat, median_stat, \
        p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
        asymmetric_error, bin_median = get_binned_stats(x, y, n)

        results = pd.DataFrame()
        results["bin_lower_edge"] = bin_edges[0:-1]
        results["bin_upper_edge"] = bin_edges[1:]
        results["bin_median"] = bin_median
        results["mean"] = mean_stat.statistic
        results["std"] = std_stat.statistic
        results["median"] = median_stat.statistic
        results["05_perc"] = p_05_stat.statistic
        results["25_perc"] = p_25_stat.statistic
        results["75_perc"] = p_75_stat.statistic
        results["95_perc"] = p_95_stat.statistic
        results["ghm"] = g

        l.append(results)

    results_df = pd.concat(l)

    return results_df


def get_binned_stats(x, y, n=11):

    # calculate binned statistics
    bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, n))
    #bin_edges = np.linspace(0, 2500, 11)
    bin_edges = np.unique(bin_edges)
    mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
    p_05_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .05), bins=bin_edges)
    p_25_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .25), bins=bin_edges)
    p_75_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .75), bins=bin_edges)
    p_95_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .95), bins=bin_edges)
    asymmetric_error = [median_stat.statistic - p_25_stat.statistic, p_75_stat.statistic - median_stat.statistic]
    bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
    #bin_median = np.linspace(125, 2375, 10)

    return bin_edges, \
           mean_stat, std_stat, median_stat, \
           p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
           asymmetric_error, bin_median


def add_regression_domains(x, y, domains, palette, **kwargs):

    from sklearn.linear_model import LinearRegression
    import matplotlib.patheffects as mpe
    outline = mpe.withStroke(linewidth=4, foreground='white')

    df = kwargs.get('data')
    df = df.dropna(subset = [x])
    df = df.dropna(subset = [y])
    df = df.dropna(subset = [domains])

    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}

    ax = plt.gca()
    slope_str = ''
    for d in df[domains].unique():
        x_tmp = df.loc[df[domains] == d, x]
        y_tmp = df.loc[df[domains] == d, y]
        reg = LinearRegression().fit(x_tmp.array.reshape(-1,1), y_tmp.array)
        #df_tmp = df.loc[df[domains] == d]
        #if x == 'Precipitation':
        #    z = 'Net radiation'
        #elif x == 'Net radiation':
        #    z = 'Precipitation'
        #r_partial_mat = partial_corr(data=df_tmp, x=x, y=y, covar=[z], method='spearman')
        #r_partial = r_partial_mat["r"].values[0]

        b0 = reg.intercept_
        b1 = reg.coef_
        #slope_str = slope_str + r' slope ' "= " + str(np.round(b1,2)) + "\n"
        x_range = np.array([np.quantile(x_tmp,0.05), np.quantile(x_tmp,0.95)])
        y_range = reg.predict(x_range.reshape(-1,1))
        ax.plot(x_range, y_range, color=palette[d], alpha=1,
                label=r' slope ' "= " + str(np.round(b1[0],2)), path_effects=[outline])

def add_corr_domains(x, y, domains, palette, **kwargs):

    df = kwargs.get('data')
    df = df.dropna(subset = [y])
    df = df.dropna(subset = [domains])

    palette = {"wet warm": '#018571', "dry warm": '#a6611a', "wet cold": '#80cdc1', "dry cold": '#dfc27d'}

    ax = plt.gca()
    corr_str = ''
    for d in df[domains].unique():
        r_sp, _ = stats.spearmanr(df.loc[df[domains] == d, x], df.loc[df[domains] == d, y], nan_policy='omit')
        df_tmp = df.loc[df[domains] == d]
        if x == 'Precipitation':
            z = 'Net radiation'
        elif x == 'Net radiation':
            z = 'Precipitation'
        #r_partial_mat = partial_corr(data=df_tmp, x=x, y=y, covar=[z], method='spearman')
        #r_partial = r_partial_mat["r"].values[0]

        corr_str = corr_str + r' $\rho_s$ ' "= " + str(np.round(r_sp,2)) + "\n"
        ax.scatter(10000, 10000, marker='o', c=palette[d], alpha=1, s=25, lw=0, label=r' $\rho_s$ ' "= " + str(np.round(r_sp,2)))

def add_corr(x, y, **kws):
    r_sp, _ = stats.spearmanr(x, y, nan_policy='omit')
    ax = plt.gca()
    ax.annotate("rho_s humid: {:.2f}".format(r_sp), xy=(.02, .95), xycoords=ax.transAxes, fontsize=10)


def plot_latitudinal_averages_rad(xlim=[-100, 2900], x_name='flux', x_unit=' [mm/year]', **kwargs):

    df = kwargs["data"]
    # specify names, units, and axes limits
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim([-60, 90])
    ax.set_xlabel(x_name + x_unit)
    ax.set_ylabel('lat [deg]')

    # calculate averages
    from helpers.avg_group import mean_group
    latavg, netradavg = mean_group(df["lat"].values, df["netrad"].values)
    latavg, petavg = mean_group(df["lat"].values, df["potevap"].values)

    # plot latitudinal averages
    ax.plot(netradavg, latavg, c='tab:green', label='netrad', alpha=0.8) #, mfc=nextcolor
    ax.plot(petavg, latavg, c='tab:orange', label='potevap', alpha=0.8) #, mfc=nextcolor


def plot_latitudinal_averages_forcing(xlim=[-100, 2600], x_name='flux', x_unit=' [mm/year]', **kwargs):

    df = kwargs["data"]
    # specify names, units, and axes limits
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim([-60, 90])
    ax.set_xlabel(x_name + x_unit)
    ax.set_ylabel('lat [deg]')

    # calculate averages
    from helpers.avg_group import mean_group
    latavg, netradavg = mean_group(df["lat"].values, df["netrad"].values)
    latavg, pavg = mean_group(df["lat"].values, df["pr"].values)

    # plot latitudinal averages
    ax.plot(netradavg, latavg, c='tab:orange', label='Net radiation', alpha=0.8) #, mfc=nextcolor
    ax.plot(pavg, latavg, c='tab:grey', label='Precipitation', alpha=0.8) #, mfc=nextcolor

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)


def plot_latitudinal_averages_fluxes(xlim=[-100, 2600], x_name='flux', x_unit=' [mm/year]', **kwargs):

    df = kwargs["data"]
    # specify names, units, and axes limits
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim([-60, 90])
    ax.set_xlabel(x_name + x_unit)
    ax.set_ylabel('lat [deg]')

    # calculate averages
    from helpers.avg_group import mean_group
    latavg, evapavg = mean_group(df["lat"].values, df["evap"].values)
    latavg, qravg = mean_group(df["lat"].values, df["qr"].values)
    latavg, qtotavg = mean_group(df["lat"].values, df["qtot"].values)

    # plot latitudinal averages
    ax.plot(evapavg, latavg, c='tab:green', label='Actual evapotranspiration', alpha=0.8) #, mfc=nextcolor
    ax.plot(qravg, latavg, c='tab:purple', label='Groundwater recharge', alpha=0.8) #, mfc=nextcolor
    ax.plot(qtotavg, latavg, c='tab:blue', label='Total runoff', alpha=0.8) #, mfc=nextcolor


def mask_greenland(data_path="2b/aggregated/"):
    ax = plt.gca()
    df_greenland = pd.read_csv(data_path + "greenland.csv", sep=',')  # greenland mask for plot
    ax.scatter(df_greenland['lon'], df_greenland['lat'], transform=ccrs.PlateCarree(),
               marker='s', s=.35, edgecolors='none', c='whitesmoke')


def plot_map(lon, lat, var, var_unit=" ", var_name="misc",
             bounds=np.linspace(0, 2000, 11), colormap='YlGnBu', colortype='Sequential', colormap_reverse=False):

    # prepare colour map
    o = brewer2mpl.get_map(colormap, colortype, 9, reverse=colormap_reverse)
    c = o.mpl_colormap

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    sc = ax.scatter(lon, lat, c=var, cmap=c, marker='s', s=.35, edgecolors='none',
                    norm=customnorm, transform=ccrs.PlateCarree())
    #ax.coastlines(linewidth=0.5)

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5, extend = 'max')
    cbar.set_label(var_name + var_unit)
    # cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
    cbar.ax.tick_params(labelsize=12)
    plt.gca().spines['geo'].set_visible(False)

    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])


def plot_outliers(df, g, var_name, var_unit=" "):

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    df_m = df[df["ghm"] == g]

    sc = ax.scatter(df_m['lon'].mask(df_m[var_name]<=df_m["Precipitation"]), df_m['lat'].mask(df_m[var_name]<=df_m["Precipitation"]),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var>precip')
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] < 10000), df_m['lat'].mask(df_m[var_name] < 10000),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var>10000 (large)')
    sc = ax.scatter(df_m['lon'].mask((df_m[var_name] > 1) | (df_m[var_name] <= 0)), df_m['lat'].mask((df_m[var_name] > 1) | (df_m[var_name] <= 0)),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='0<var<=1 (small)')
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] != 0), df_m['lat'].mask(df_m[var_name] != 0),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var=0 (zero)')
    sc = ax.scatter(df_m['lon'].mask((df_m[var_name] >= 0)), df_m['lat'].mask((df_m[var_name] >= 0)),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var<0 (negative)')
    df_m[var_name][np.isnan(df_m[var_name])] = -999
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] != -999), df_m['lat'].mask(df_m[var_name] != -999),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var=nan',
                    facecolor='tab:grey')

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    plt.gca().spines['geo'].set_visible(False)

    ax.set_title(g, fontsize=10)
    leg = ax.legend(loc='lower left', markerscale=4, fontsize=7)
    leg.set_title(title="var = " + var_name + var_unit, prop={'size': 8})

    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])


def plot_most_deviating_model(df, max_ind, ghms, var_name):

    o = brewer2mpl.get_map("Set1", 'Qualitative', 9, reverse=True)
    c = o.mpl_colormap

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    bounds = np.linspace(-1.5, len(ghms) - 0.5, len(ghms) + 2)
    customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    sc = ax.scatter(df['lon'], df['lat'], transform=ccrs.PlateCarree(), norm=customnorm,
                    marker='s', s=.35, edgecolors='none', c=np.ma.masked_equal(max_ind, -2), cmap=c)

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.75, aspect=30)
    cbar.set_label('Most deviating model' + " (" + var_name + ")")
    # cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.tick_params(rotation=45)
    plt.gca().spines['geo'].set_visible(False)

    cbar.set_ticks(np.arange(-1, len(ghms), 1))
    cbar.set_ticklabels(["multiple"] + ghms)

    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

