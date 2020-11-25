import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

folder = '/projects/NS9600K/shofer/blowing_snow/observations/'

# Open up all the data
LWD_Y_no = pd.read_csv(folder + 'Out_YYR_LD_o.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWD_Y_bs = pd.read_csv(folder + 'Out_YYR_LD_a.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWD_Y_no = pd.read_csv(folder + 'Out_YYR_SD_o.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWD_Y_bs = pd.read_csv(folder + 'Out_YYR_SD_a.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWU_Y_no = pd.read_csv(folder + 'Out_YYR_LU_o.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
LWU_Y_bs = pd.read_csv(folder + 'Out_YYR_LU_a.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWU_Y_no = pd.read_csv(folder + 'Out_YYR_SU_o.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
                                                                                                     'Mean bias', 'Mean obs', 'RMSE', 'Centered RMSE', 'correlation',  'obs std', 'mar std'])
SWU_Y_bs = pd.read_csv(folder + 'Out_YYR_SU_a.dat', sep='\s+', engine='python', skipfooter=1, names=['Period', 'Station', 'Var',
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

df_mb = pd.DataFrame(
    final_arrs[0], columns=list_cols, index=list(df.Station.values))
df_mb.index = df_mb.index.rename('Station')

df_rmse = pd.DataFrame(
    final_arrs[1], columns=list_cols, index=list(df.Station.values))

df_correlation = pd.DataFrame(
    final_arrs[2], columns=list_cols, index=list(df.Station.values))

df_weights = pd.DataFrame(arr_weights, columns=[
                          list_cols], index=list(points.Station))
df_weights.index = df_weights.index.rename('Station')
df_weights.to_csv('WEIGHTS.csv')


df_final = pd.DataFrame(
    arr, columns=[list_cols], index=list(points.Station))
df_final.index = df_final.index.rename('Station')
# To get R**2 instead of R
df_final = df_final ** 2
df_final.to_csv('R2.csv')

df_rmse = pd.DataFrame(
    arr_rmse, columns=[list_cols], index=list(points.Station))
# df_rmse = df_rmse.drop(['SCO_L', 'QAS_U', 'NUK_U', 'NUK_L', 'TAS_U', 'S5', 'S6', 'S9'])
# df_rmse = df_rmse.drop(['SCO_L', 'QAS_U', 'NUK_U', 'NUK_L', 'TAS_U', 'MIT', 'KAN_B'])
df_rmse.index = df_rmse.index.rename('Station')
df_rmse.to_csv('RMSE.csv')
df_rmse_final = pd.DataFrame(df_rmse.mean().values.reshape(
    1, 8), columns=[list_cols], index=['RMSE'])
df_rmse.to_csv('RMSE_mean.csv')

df_mb = pd.DataFrame(
    arr_mb, columns=[list_cols], index=list(points.Station))
# df_mb = df_mb.drop(['SCO_L', 'QAS_U', 'NUK_U', 'NUK_L', 'TAS_U', 'S5', 'S6', 'S9'])
# df_mb = df_mb.drop(['SCO_L', 'QAS_U', 'NUK_U', 'NUK_L', 'TAS_U', 'MIT', 'KAN_B'])
# df_mb = df_mb.drop(['SCO_L', 'QAS_U', 'NUK_U', 'NUK_L', 'TAS_U', 'S5', 'S6', 'S9', 'MIT', 'KAN_B'])

df_mb.index = df_mb.index.rename('Station')
df_mb.to_csv('MB.csv')
df_mb_final = pd.DataFrame(df_mb.mean().values.reshape(
    1, 8), columns=[list_cols], index=['Mean bias'])
df_mb_final.to_csv('MB_mean.csv')
