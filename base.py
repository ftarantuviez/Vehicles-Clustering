import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering 

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Clustering on Vehicle dataset', page_icon="./f.png")
st.title('Clustering on Vehicle dataset')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
Imagine that an automobile manufacturer has developed prototypes for a new vehicle. Before introducing the new model into its range, the manufacturer wants to determine which existing vehicles on the market are most like the prototypes--that is, how vehicles can be grouped, which group is the most similar with the model, and therefore which models they will be competing against.

Our objective here, is to use clustering methods, to find the most distinctive clusters of vehicles. It will summarize the existing vehicles and help manufacturers to make decision about the supply of new models.
""")
pdf = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/cars_clus.csv")

st.write("## About dataset")
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)

st.dataframe(pdf)

st.write(""" 
## Feature Selection

The features selected to fit the model were: the engine size, horsepower, wheel bas, length, curb_wgt, fuel capacity and mpg.
""")

featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
st.dataframe(pdf)

st.write("## Clustering")

st.sidebar.header("Sidebar")
st.sidebar.write("Algorithm")
method = st.sidebar.selectbox("Solve using: ", ["Scikit-learn", "Scipy"])


x = featureset.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
leng = feature_mtx.shape[0]

if method == "Scikit-learn":
  dist_matrix = euclidean_distances(feature_mtx,feature_mtx) 
  Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
  fig = pylab.figure(figsize=(18,50))
  Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')

  fig = pylab.figure(figsize=(18,50))
  def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
  dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =16, orientation = 'right')
  st.pyplot(fig)
else:  
  dist_matrix = np.zeros([leng,leng])
  for i in range(leng):
    for j in range(leng):
      dist_matrix[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
  Z = hierarchy.linkage(dist_matrix, 'complete')
  max_d = 3
  clusters = fcluster(Z, max_d, criterion='distance')
  k = 5
  clusters = fcluster(Z, k, criterion='maxclust')

  def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
  
  plt.rcParams['axes.facecolor'] = '#292929'
  fig = pylab.figure(figsize=(18,50))
  dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =25, orientation = 'right')
  if st.button("Save Image"):
    plt.savefig("dendogram_scipy.png")
  st.pyplot(fig)

st.write("""
Now, we can use the 'AgglomerativeClustering' to cluster the dataset. The AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. The linkage criteria determines the metric used for the merge strategy:

-   Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
-   Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
-   Average linkage minimizes the average of the distances between all observations of pairs of clusters.

""")
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(dist_matrix)
pdf['cluster_'] = agglom.labels_

n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.rcParams['axes.facecolor'] = '#fff'
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
  subset = pdf[pdf.cluster_ == label]
  for i in subset.index:
    plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
  plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

st.pyplot()

st.write("""
As you can see, we are seeing the distribution of each cluster using the scatter plot, but it is not very clear where is the centroid of each cluster. Moreover, there are 2 types of vehicles in our dataset, "truck" (value of 1 in the type column) and "car" (value of 1 in the type column). So, we use them to distinguish the classes, and summarize the cluster. First we count the number of cases in each group:
""")

st.write("Now we can look at the characteristics of each cluster:")
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
st.dataframe(agg_cars)
st.write("""
It is obvious that we have 3 main clusters with the majority of vehicles in those.

**Cars**:

-   Cluster 1: with almost high mpg, and low in horsepower.
-   Cluster 2: with good mpg and horsepower, but higher price than average.
-   Cluster 3: with low mpg, high horsepower, highest price.

**Trucks**:

-   Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
-   Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
-   Cluster 3: with good mpg and horsepower, low price.

Please notice that we did not use **type** , and **price** of cars in the clustering process, but Hierarchical clustering could forge the clusters and discriminate them with quite high accuracy.

""")

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
  subset = agg_cars.loc[(label,),]
  for i in subset.index:
    plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
  plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

st.pyplot()


# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/)TODO
""")
# / This app repository