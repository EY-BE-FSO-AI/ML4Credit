""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""
#Implementation of the clustering
""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

development_set["Dflt_year"] = development_set.Default_date.astype("datetime64[ns]").dt.year
#Select relevant data
development_set["pd_q"] = pd.qcut(development_set.PD.dropna(), q = 40, labels=np.arange(0,40))
Data = development_set[["PD", "Default_Binary","Dflt_year","pd_q"]].dropna()
Data_q = Data.groupby(["pd_q", "Dflt_year"]).Default_Binary.mean()
Data_q.unstack().plot(marker='o', linestyle='None') #Or Data_q.unstack().plot(stacked=True, marker='o', linestyle='None')

#KMeans
# --> Attention nan values in 2007 for some deciles:
Data_q = Data_q.unstack().fillna( 0 )
Data_q.plot(marker='o', linestyle='None')
#Visually 
model = KMeans(n_clusters=7)
model.fit(Data_q)
Data_q["cluster_num"] = model.labels_
grade_dict = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G"}
cluster_dict = Data_q.to_dict()["cluster_num"]
grade_map = {k : grade_dict[v] for k, v in cluster_dict.items()} #{Pd percentile : cluster grade}

development_set["cluster_label"] = development_set.pd_q.apply(lambda g : grade_map[g] )


""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""