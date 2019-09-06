import os
import pandas                      as pd
import numpy                       as np
import scipy.stats                 as stat
import matplotlib.pyplot           as plt
import scipy.optimize              as optimize
from scipy.special import loggamma as lgamma

os.chdir("..")    #Move back to the src folder
# Load the data
df = pd.read_csv(os.getcwd()+r'\features\LGD_model_dataset.csv')

# Optimize linear model
def ll(data, b):
  y   = data[0]
  x1   = data[1]
  x2   = data[2]
  x3   = data[3]
  mu  = np.exp(b[0]+b[1]*x1) / (1+np.exp(b[0]+b[1]*x1))
  p0  = np.exp(b[2]+b[3]*x2) / (1+np.exp(b[2]+b[3]*x2))
  p1  = np.exp(b[4]+b[5]*x3) / (1+np.exp(b[4]+b[5]*x3))
  phi = np.exp(b[6])
  if y==0:
    ll   = np.log(p0)  
  elif y==1:
    ll   = np.log(p1)
  else:
    ll   = lgamma(phi) - lgamma(mu*phi) - lgamma((1-mu)*phi) + (mu*phi-1)*np.log(y) + (phi-mu*phi-1)*np.log(1-y) + np.log(1-p0) + np.log(1-p1)
  return -ll
def mll(B, data):
  df= data[['Y', 'LTV_LR', 'LTV_0', 'LTV_1']]
  return df.apply(lambda DATA: ll(DATA, B), axis=1).sum()

for i in df.segment.unique():
  solution                          = optimize.minimize(fun=mll,x0=[-1,0,-1,0,-1,0,1],args=df[df.segment==i],method='SLSQP',bounds=((-30,30),(-30,30),(-4,4),(-4,4),(-4,4),(-4,4),(0.1,None)))
  df.loc[df.segment==i, 'mu']       = np.exp(solution.x[0]+solution.x[1]*df.LTV_LR)/(1+np.exp(solution.x[0]+solution.x[1]*df.LTV_LR))
  df.loc[df.segment==i, 'p1']       = np.exp(solution.x[2]+solution.x[3]*df.LTV_0)/(1+np.exp(solution.x[2]+solution.x[3]*df.LTV_0))
  df.loc[df.segment==i, 'p0']       = np.exp(solution.x[4]+solution.x[5]*df.LTV_1)/(1+np.exp(solution.x[4]+solution.x[5]*df.LTV_1))
  df.loc[df.segment==i, 'LGD_hat']  = (1-df.p1)*(1-df.p0)*df.mu+df.p1
  print(solution)
  plt.show(plt.plot(df.loc[df.segment==i, 'LGD_hat']))

plt.show(plt.scatter(df.LGD_hat, df.Y))

# Output Classic LGD model


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