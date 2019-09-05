import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats       as stat
import seaborn           as sns

os.chdir("..")    #Move back to the src folder
# Load the data
df = pd.read_csv(os.getcwd()+r'\data\lgd.csv', sep=",")

# Rename column
df['Y'] = df.lgd_time.apply(lambda x: 0 if x<0.001 else 1 if x>0.999 else x)

# Plot distribution and features
for i in ['Y', 'LTV', 'event', 'purpose1']:
    sns.distplot(df[i], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})
    plt.show()

# Output Y dataset and X dataset
df.Y.to_csv(os.getcwd()+r'\data\Y.csv')
df[['LTV', 'event', 'purpose1']].to_csv(os.getcwd()+r'\data\X.csv')

# Compare Kernel Density with the calibrated Beta Density
fit_k   = stat.gaussian_kde(df.Y) 
fit_b   = stat.beta.fit(df.Y[(df.Y<1) & (df.Y>0)], floc=0, fscale=1)
kernel  = df.Y.map(lambda x: fit_k(x)[0])
beta    = df.Y.map(lambda x: stat.beta.pdf(x, fit_b[0], fit_b[1], loc=fit_b[2], scale=fit_b[3]))
plot    = pd.DataFrame({'Y': df.Y, 'kernel': list(kernel)}).sort_values(by=['Y','kernel']).drop_duplicates()
fig, ax = plt.subplots(facecolor=(.5, .5, .5))
ax.set_facecolor('#DCDCDC')
ax.set_title('Kernel PDF vs. Beta PDF', color='0.8')
ax.set_xlabel('LGD', color='Black')
ax.set_ylabel('PDF', color='Orange')
ax.plot(plot.Y, plot.kernel, label='Kernel')
ax.scatter(df.Y, beta, color='Red', marker="o", label='Beta')
ax.legend(loc='upper middle')
ax.tick_params(labelcolor='tab:blue')
ax.set_yscale('log')
plt.show()