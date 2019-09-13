import os
import pandas                      as pd
import numpy                       as np
import scipy.stats                 as stat
import matplotlib.pyplot           as plt
import scipy.optimize              as optimize
import sklearn.linear_model        as ln
from scipy.special import loggamma as lgamma

os.chdir("..")    #Move back to the src folder
# Load the data
df = pd.read_csv(os.getcwd()+r'\features\LGD_model_dataset.csv')

# Optimize linear model
def ll(data, b):
  y   = data[0]
  x1  = data[1]
  x2  = data[2]
  x3  = data[3]
  p0  = np.exp(b[0]+b[1]*x2) / (1+np.exp(b[0]+b[1]*x2))
  mu  = np.exp(b[2]+b[3]*x1) / (1+np.exp(b[2]+b[3]*x1))
  p1  = np.exp(b[4]+b[5]*x3) / (1+np.exp(b[4]+b[5]*x3))
  phi = np.exp(b[6])
  if y==0:
    ll   = np.log(p0)  
  elif y==1:
    ll   = np.log(p1)
  else:
    ll   = lgamma(phi) - lgamma(mu*phi) - lgamma(phi-mu*phi) + (mu*phi-1)*np.log(y) + (phi-mu*phi-1)*np.log(1-y) + np.log(1-p0) + np.log(1-p1)
  return -ll
def mll(B, data):
  df= data[['Y', 'LTV_LR', 'LTV_0', 'LTV_1']]
  return df.apply(lambda DATA: ll(DATA, B), axis=1).sum()

for i in df.segment.unique():
  if (df.Y[df.segment==i].isin([1]).sum()==0) | (df.Y[df.segment==i].isin([0]).sum()==0):
    o1 = 0; o2 = 0; o5 = 0; o6 = 0
  else:
    lr    = ln.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(df[['LTV_0','LTV_1']][(df.segment==i) & (df.Y.isin([0,1]))], df.Y[(df.segment==i) & (df.Y.isin([0,1]))])
  o1    = pd.DataFrame(lr.coef_)[0]
  o2    = lr.intercept_
  o5    = pd.DataFrame(lr.coef_)[1]
  o6    = lr.intercept_
  dummy = df.Y[(df.segment==i) & (~df.Y.isin([0,1]))]
  if dummy.count()>0:
  	lm    = ln.LinearRegression().fit(df.LTV_LR[(df.segment==i) & (~df.Y.isin([0,1]))].values.reshape(-1,1), np.log(dummy/(1-dummy)))
  	o3    = lm.coef_
  	o4    = lm.intercept_
  else:
        o3=0; o4=0;
  solution                          = optimize.minimize(fun=mll,x0=[o1,o2,o3,o4,o5,o6,-1],args=df[(df.segment==i) & (df.training==1)],method='L-BFGS-B',bounds=((None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(-1,2)))
  df.loc[df.segment==i, 'mu']       = np.exp(solution.x[2]+solution.x[3]*df.LTV_LR)/(1+np.exp(solution.x[2]+solution.x[3]*df.LTV_LR))
  df.loc[df.segment==i, 'p0']       = np.exp(solution.x[0]+solution.x[1]*df.LTV_0)/(1+np.exp(solution.x[0]+solution.x[1]*df.LTV_0))
  df.loc[df.segment==i, 'p1']       = np.exp(solution.x[4]+solution.x[5]*df.LTV_1)/(1+np.exp(solution.x[4]+solution.x[5]*df.LTV_1))
  df.loc[df.segment==i, 'LGD_hat']  = (1-df.p1)*(1-df.p0)*df.mu+df.p1
  print(solution)
  plt.show(plt.plot(df.loc[df.segment==i, 'LGD_hat']))

plt.show(plt.scatter(df.LGD_hat, df.Y))

# Output Classic LGD model
print(stat.spearmanr(df.Y,df.mu))
print(stat.spearmanr(df.Y,df.LGD_hat))
df['deciles']=np.around(df.LGD_hat,decimals=1)
print(df.groupby('deciles')['Y'].mean())

# Plot joint distribution LGD and LTV

# Extract x and y
x = df.LTV
y = df.Y
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# Kernel 
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stat.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
# plot
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show(plt.title('2D Gaussian Kernel density estimation'))

df.to_csv(os.getcwd()+r'\models\LGD_model_dataset_output.csv')