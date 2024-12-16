# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import OrdinalEncoder

# %%
with open('imports-85.names', 'r') as file: 
    content = file.read() 
    
print(content)

# %%
headers = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_doors', 'body_style', 'drive_wheels', 'engine_location', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
df = pd.read_csv('imports-85.data', names=headers)
df

# %% [markdown]
# ### replace ? to nan

# %%
df.replace('?', np.nan, inplace=True)

# %% [markdown]
# ### Impute nulls

# %%
nulls_col = df.columns[df.isnull().sum() > 0]
nulls_col = df.columns

# %%
# Separate numeric and categorical features
numeric_features = df.select_dtypes(exclude='object')
categorical_features = df.select_dtypes(include='object')

# %%
numeric_features = [feat for feat in nulls_col if df[feat].dtype.kind in 'bifc'] 
categorical_features = [feat for feat in nulls_col if feat not in numeric_features]

# %%
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])


categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

# %%
df.isna().sum().sum()

# %% [markdown]
# ### Scale 

# %%
mms = MinMaxScaler() 
x_scaled = mms.fit_transform(df[['price','horsepower']])

# %%
x_scaled

# %% [markdown]
# ### Elbow, silhouette_score, calinski_harabasz_score

# %%
inertias = []
s_scores=[]
c_scores=[]

for k in range(2,11): 
    km= KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(x_scaled)

    inertias.append(km.inertia_)

    score = silhouette_score(x_scaled, km.labels_)
    s_scores.append(score)

    score = calinski_harabasz_score(x_scaled, km.labels_)
    c_scores.append(score)


# %%
plt.plot(range(2,11), inertias);

# %%
plt.plot(range(2,11), s_scores);

# %%
plt.plot(range(2,11), c_scores);

# %%
optimal_k = 3 
km = KMeans(n_clusters=optimal_k, random_state=42) 
km.fit(x_scaled) 

# %%
labels = km.labels_ 
labels

# %%
centers =km.cluster_centers_
centers

# %%
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=km.labels_);
plt.scatter(centers[:,0], centers[:,1], c='r', marker='X');

# %%
vor = Voronoi(centers)

fig, ax = plt.subplots(figsize=(8, 6))

voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2);

ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=km.labels_, cmap='viridis', s=50, alpha=0.6, edgecolor='k'); 
ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolor='k');
ax.set_xlabel('Horsepower') 
ax.set_ylabel('Price') 
ax.set_title('Voronoi Diagram of KMeans Clusters')

# %% [markdown]
# ##### boundaries (method 2)

# %%
from matplotlib.colors import ListedColormap 
h = 0.02 #step size for the mesh grid

# Create custom color maps for the plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) # for the background of the decision boundaries.
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # for the scatter plot of the data points.

# range of the mesh grid based on the scaled data.
x_min, x_max = x_scaled[:, 0].min() - 1, x_scaled[:, 0].max() + 1 
y_min, y_max = x_scaled[:, 1].min() - 1, x_scaled[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  #function creates a mesh grid for plotting, generates 2D arrays representing the coordinates of the grid points.
                     np.arange(y_min, y_max, h)) 

# Predict cluster labels for each point in the mesh 
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])  # .ravel() - flatten the xx and yy arrays into 1D
Z = Z.reshape(xx.shape) # reshapes the predicted labels to match the shape of the mesh grid

plt.figure(figsize=(8, 6)) 

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5) # fills the contours of the mesh grid with colors
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=labels, cmap=cmap_bold, s=50) 
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X') 

plt.xlabel('Horsepower') 
plt.ylabel('Price') 
plt.title('KMeans Clustering with Decision Boundaries') 

# %% [markdown]
# ### DBSCAN

# %%
dbs = DBSCAN(eps = 0.15, min_samples = 5) 
dbs.fit(x_scaled)

# %%
plt.scatter(x_scaled[:,0], x_scaled[:,1], c=dbs.labels_); 

# %% [markdown]
# ### Nearest-Neighbors

# %%
df_cleaned = pd.read_csv('mlb_batting_cleaned.csv')
df_cleaned

# %%
name = df_cleaned['Name']
X = df_cleaned.drop(columns='Name')

# %%
name

# %%
categorical_features = X.select_dtypes(include='object').columns
categorical_features

# %%
X['Lg'] = np.where(X['Lg'] == 'NL',1,0)

# %%
oe = OrdinalEncoder()
oe.fit(X[['Tm']])
X['Tm'] = oe.transform(X[['Tm']])

# %%
numeric_features = X.select_dtypes(exclude='object').columns
numeric_features

# %%
scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(X)

# %%
def nearest_neighbors(x_scaled, player_name, name_series):
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(x_scaled)

     
    player_index = name_series[name_series == player_name].index[0]

    
    dist, neighbors = nn.kneighbors([x_scaled[player_index]])

    closest_players = name_series.iloc[neighbors[0][1:]].values

    print(f"Input player name: {player_name}")
    print(f"the first closest player: {closest_players[0]}")
    print(f"the second closest player: {closest_players[1]}")
    

# %%
nearest_neighbors(x_scaled, 'Shohei Ohtani', name)


