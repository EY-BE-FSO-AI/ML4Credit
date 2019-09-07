import os
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import scipy.stats       as stat    

os.chdir("..")    #Move back to the src folder
# Load the data
df1        = pd.DataFrame({'Y': pd.read_csv(os.getcwd()+"\data\Y.csv").iloc[:,1]})
df2        = pd.read_csv(os.getcwd()+"\data\X.csv")
df         = df1.join(df2)
df['Y_Z']  = stat.norm.ppf(stat.rankdata(df.Y)/(len(df.Y)+1))

# Plot distribution Y per segment
for i in ['event', 'purpose1']:
  dummy   = df[i].unique()
  fig, ax = plt.subplots(facecolor=(.5, .5, .5))
  ax.set_facecolor('#DCDCDC')
  ax.set_title('LGD PDF: segmentation variable '+i+' '+str(dummy[0])+' vs '+str(dummy[1]), color='0.8')
  for j in dummy:
    sns.distplot(df.Y[df[i]==j], hist=False, rug=True, label=str(j))
  ax.legend(loc='upper center')
  ax.tick_params(labelcolor='tab:blue')
  plt.show()

# Transform features and plot against the Y axis
df['quantile'] = df.LTV
df['deciles']  = df.LTV
df['deciles_'] = df.LTV
df['Y0']       = df.LTV
df['Y1']       = df.LTV
df['LTV_Z']    = df.LTV
df['segment']  = df.LTV
n              = 1
for i in df.event.unique():
  for j in df.purpose1.unique():
    df.loc[(df.purpose1==i) & (df.event==j), 'quantile'] = stat.rankdata(df.loc[(df.purpose1==i) & (df.event==j), 'LTV'])/(len(df.loc[(df.purpose1==i) & (df.event==j), 'LTV'])+1)
    df.loc[(df.purpose1==i) & (df.event==j), 'deciles']  = np.round(np.around(df.loc[(df.purpose1==i) & (df.event==j), 'quantile'], decimals=2)/0.05)*0.05
    df.loc[(df.purpose1==i) & (df.event==j), 'deciles_'] = np.around(df.loc[(df.purpose1==i) & (df.event==j), 'quantile'], decimals=1)
    df.loc[(df.purpose1==i) & (df.event==j), 'Y0']       = df.loc[(df.purpose1==i) & (df.event==j), 'Y']==0
    df.loc[(df.purpose1==i) & (df.event==j), 'Y1']       = df.loc[(df.purpose1==i) & (df.event==j), 'Y']==1
    df.loc[(df.purpose1==i) & (df.event==j), 'LTV_Z']    = stat.norm.ppf(df.loc[(df.purpose1==i) & (df.event==j), 'quantile'])
    df.loc[(df.purpose1==i) & (df.event==j), 'segment']  = n
    plt.scatter(df.loc[(df.purpose1==i) & (df.event==j), 'LTV_Z'], df.loc[(df.purpose1==i) & (df.event==j), 'Y_Z'])
    plt.show()
    n += 1
df = pd.merge(df, np.log((df.groupby(['segment','deciles_'])['Y0'].count()+1-df.groupby(['segment','deciles_'])['Y0'].sum())/(df.groupby(['segment','deciles_'])['Y0'].sum()+1)) \
     .reset_index(name='LTV_0'), on=['segment', 'deciles_'], how='inner')
df = pd.merge(df, np.log((df.groupby(['segment','deciles_'])['Y1'].count()+1-df.groupby(['segment','deciles_'])['Y1'].sum())/(df.groupby(['segment','deciles_'])['Y1'].sum()+1)) \
     .reset_index(name='LTV_1'), on=['segment', 'deciles_'], how='inner')
df = pd.merge(df, np.log((df.groupby(['segment','deciles'])['Y'].mean() + 0.0001)/(1-(df.groupby(['segment','deciles'])['Y'].mean() + 0.0001))) \
     .reset_index(name='LTV_LR'), on=['segment', 'deciles'], how='inner')


# Output model dataset
df[['Y', 'Y_Z', 'LTV', 'LTV_Z', 'LTV_LR', 'LTV_1', 'LTV_0', 'segment']].to_csv(os.getcwd()+r'\features\LGD_model_dataset.csv')