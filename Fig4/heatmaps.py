import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


folder = '/projects/NS9600K/shofer/blowing_snow/observations/new/'

# Open up all the data
LWD_Y_no = pd.read_csv(folder + 'Out_YYR_LD_MARv3.11-ERA5.dat', sep='\s+', engine='python',
                       skipfooter=1, names=['Period', 'Station', 'Var',
                                            'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWD_Y_bs = pd.read_csv(folder + 'Out_YYR_LD_MARv3.11-BS.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                               'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWD_Y_no = pd.read_csv(folder + 'Out_YYR_SD_MARv3.11-ERA5.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                                 'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWD_Y_bs = pd.read_csv(folder + 'Out_YYR_SD_MARv3.11-BS.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                               'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWU_Y_no = pd.read_csv(folder + 'Out_YYR_LU_MARv3.11-ERA5.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                                 'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWU_Y_bs = pd.read_csv(folder + 'Out_YYR_LU_MARv3.11-BS.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                               'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWU_Y_no = pd.read_csv(folder + 'Out_YYR_SU_MARv3.11-ERA5.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                                 'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWU_Y_bs = pd.read_csv(folder + 'Out_YYR_SU_MARv3.11-BS.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                               'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
# Create a list with the data files for looping
list_data = [LWD_Y_no, LWD_Y_bs, SWD_Y_no, SWD_Y_bs,
             LWU_Y_no, LWU_Y_bs, SWU_Y_no, SWU_Y_bs]
# Create a list such as ['SWD_R', 'SWD_M', 'LWD_R', ...]
list1 = [var + '' for var in ['LWD', 'SWD', 'LWU', 'SWU']]
list2 = [var + '_bs' for var in ['LWD', 'SWD', 'LWU', 'SWU']]
cols = [x for x in itertools.chain.from_iterable(
    itertools.zip_longest(list1, list2)) if x]
list_cols = [x for x in itertools.chain.from_iterable(
    itertools.zip_longest(list1, list2)) if x]
# Add 'Station' to first position of the list
cols.insert(0, 'Station')


# Extract all the radiation values and put them in one array
# This is preparation work for the heatmaps
final_arrs = []
for var in ['Mean bias', 'RMSE', 'correlation']:
    arr = np.zeros((19, 8))
    i = 0
    for df in list_data:
        try:
            df = df[df.Station != 'AWS1_NARE9697_s']
        except:
            pass
        try:
            df = df[df.Station != 'AWS3_NARE9697_s']
        except:
            pass

        arr[:, i] = df[var].values

        print("i is {}".format(i))
        i += 1
    final_arrs.append(arr)

cols_final = ['LWD', 'SWD', 'LWU', 'SWU']

df_mb = pd.DataFrame(
    final_arrs[0], columns=list_cols, index=list(df.Station.values))
df_mb.index = df_mb.index.rename('Station')
# For percentage changes this works (WIP)
df_mb_perc = (-1) * ((abs(df_mb.mean().iloc[[0, 2, 4, 6]].values) - abs(df_mb.mean().iloc[[
    1, 3, 5, 7]])) / abs(df_mb.mean().iloc[[0, 2, 4, 6]].values)) * 100
# For absolute changes in Wm-2 this works
df_mb_abs = (-1) * (abs(df_mb.mean().iloc[[0, 2, 4, 6]].values) -
                    abs((df_mb.mean().iloc[[1, 3, 5, 7]])))


df_mb_perc_final = pd.DataFrame(df_mb_perc.values.reshape(
    1, 4), columns=cols_final, index=[r'$\Delta$ Mean bias (%)'])

df_mb_abs_final = pd.DataFrame(df_mb_abs.values.reshape(
    1, 4), columns=cols_final, index=[r'$\Delta$ Mean bias $(Wm^{-2})$'])


df_rmse = pd.DataFrame(
    final_arrs[1], columns=list_cols, index=list(df.Station.values))
df_rmse.index = df_rmse.index.rename('Station')
df_rmse_perc = (-1) * ((abs(df_rmse.mean().iloc[[0, 2, 4, 6]].values) - abs(df_rmse.mean().iloc[[
    1, 3, 5, 7]])) / abs(df_rmse.mean().iloc[[0, 2, 4, 6]].values)) * 100
df_rmse_abs = (-1) * (abs(df_rmse.mean().iloc[[0, 2, 4, 6]].values) -
                      abs((df_rmse.mean().iloc[[1, 3, 5, 7]])))

df_rmse_perc_final = pd.DataFrame(df_rmse_perc.values.reshape(
    1, 4), columns=cols_final, index=[r'$\Delta$ RMSE (%)'])

df_rmse_abs_final = pd.DataFrame(df_rmse_abs.values.reshape(
    1, 4), columns=cols_final, index=[r'$\Delta$ RMSE'])

df_correlation = pd.DataFrame(
    final_arrs[2], columns=list_cols, index=list(df.Station.values))
df_correlation.index = df_correlation.index.rename('Station')
df_correlation_perc = (-1) * ((abs(df_correlation.mean().iloc[[0, 2, 4, 6]].values) - abs(df_correlation.mean().iloc[[
    1, 3, 5, 7]])) / abs(df_correlation.mean().iloc[[0, 2, 4, 6]].values)) * 100
df_correlation_abs = (-1) * (abs(df_correlation.mean().iloc[[0, 2, 4, 6]].values) -
                             abs((df_correlation.mean().iloc[[1, 3, 5, 7]])))
# FROM HERE DOWN COPIED FROM DIFFERENT SCRIPT
figg, axx = plt.subplots(
    nrows=3, ncols=1, figsize=(10, 3), sharex=True)
h = sns.heatmap(df_mb_abs_final, annot=True,
                ax=axx[0], cmap='RdBu_r', center=0, annot_kws={"size": 12})
j = sns.heatmap(df_mb_perc_final, annot=True,
                ax=axx[1], cmap='RdBu_r', center=0, annot_kws={"size": 12})
k = sns.heatmap(df_rmse_abs_final, annot=True,
                ax=axx[2], cmap='RdBu_r', center=0, annot_kws={"size": 12})
h.set_yticklabels(h.get_yticklabels(), rotation=0)
j.set_yticklabels(j.get_yticklabels(), rotation=0)
k.set_yticklabels(k.get_yticklabels(), rotation=0)
for ax in axx:
    ax.set_xlabel('')
figg.tight_layout()
figg.savefig('/tos-project2/NS9600K/shofer/blowing_snow/heatmap.png')

# Plot for all the stations mean bias
figg, axx = plt.subplots(
    nrows=1, ncols=1, figsize=(7, 7), sharex=True)
h = sns.heatmap(df_mb, annot=True,
                ax=axx, cmap='RdBu_r', center=0, annot_kws={"size": 12})

h.set_yticklabels(h.get_yticklabels(), rotation=0)
axx.set_xlabel(r'Radiation - Mean bias $(Wm^{-2})$')
figg.tight_layout()
figg.savefig('/tos-project2/NS9600K/shofer/blowing_snow/heatmap_all_mb.png')
