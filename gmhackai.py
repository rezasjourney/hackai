
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import gmaps

key = 'AIzaSyCsr6LZaIvS7F6ZrcmEHZ3DwnU5UlRTcRo'
gmaps.configure(api_key=key)


# In[3]:

df = pd.read_json("greenPParking2015.json")


# In[14]:

factor_price = False
factor_space = False
# indoor/outdoor - weather
# traffic condition


# In[15]:

parks = df.carparks
lat = [float(p['lat']) for p in parks]
lng = [float(p['lng']) for p in parks]
price = [float(p['rate_half_hour']) for p in parks]

park_features = c = np.c_[(lat,lng)]

if factor_price:
    park_features = c = np.c_[(lat,lng, price)]
if factor_space:
    capacity = [int(p['capacity']) for p in parks]

    num_parks = len(df.index)
    usage = np.random.rand(num_parks)
    availability = capacity * usage

    max_availability = max(availability)

    park_features = c = np.c_[(lat,lng, price, max_availability - availability)]
    
X1 = np.load('X1.npy')
X2 = np.load('X2.npy')
X3 = np.load('X3.npy')
X4 = np.load('X4.npy')

park_locations = park_features[:,:2]
park_markers = gmaps.marker_layer(park_locations)
fig_plocations = gmaps.figure()
fig_plocations.add_layer(park_markers)

X1_hm = gmaps.heatmap_layer(X1)
X2_hm = gmaps.heatmap_layer(X2)
X3_hm = gmaps.heatmap_layer(X3)
X4_hm = gmaps.heatmap_layer(X4)
fig_hm = gmaps.figure()
fig_hm.add_layer(X1_hm)
fig_hm.add_layer(X2_hm)
fig_hm.add_layer(X3_hm)
fig_hm.add_layer(X4_hm)

X = np.vstack((X1, X2, X3, X4))
if factor_price:
    X = np.column_stack((X, np.zeros((len(X), 1))))
if factor_space:
    X = np.column_stack((X, np.zeros((len(X), 2))))
    
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
Xmean = kmeans.cluster_centers_

Xm = Xmean[:,:2]
Xm_sym = gmaps.symbol_layer(Xm, fill_color='black', scale=2)
fig_hm_cnt = gmaps.figure()
fig_hm_cnt.add_layer(X1_hm)
fig_hm_cnt.add_layer(X2_hm)
fig_hm_cnt.add_layer(X3_hm)
fig_hm_cnt.add_layer(X4_hm)
fig_hm_cnt.add_layer(Xm_sym)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(park_features)
distances, indices = nbrs.kneighbors(Xmean)

park_opt_id = indices.flatten()
park_opt = park_features[park_opt_id][:,:2]
opt_markers = gmaps.marker_layer(park_opt)

fig_opt = gmaps.figure()
fig_opt.add_layer(X1_hm)
fig_opt.add_layer(X2_hm)
fig_opt.add_layer(X3_hm)
fig_opt.add_layer(X4_hm)
fig_opt.add_layer(Xm_sym)
fig_opt.add_layer(opt_markers)


# In[10]:

fig_plocations


# In[11]:

fig_hm


# In[12]:

fig_hm_cnt


# In[16]:

fig_opt


# - History of parking spots
# - User's perferences - price vs convinience
# - rain
# - busy periods
