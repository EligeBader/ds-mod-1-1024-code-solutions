# %% [markdown]
# # Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import root_mean_squared_error
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold


# %% [markdown]
# # Read Data

# %%
books_data =pd.read_csv('Books.csv')
books_data
# parent_asin: Parent ID of the product

# %%
len(books_data.parent_asin.unique())

# %%
books_data.info(show_counts=True)

# %% [markdown]
# # Input ID

# %%
valid_user_ids = books_data['user_id'].to_list()

def get_valid_user_id(): 
    while True:
        uid = input("Enter user ID: ") 
        if uid in valid_user_ids: 
            return uid 
        else: 
            print("Invalid ID. Please enter a valid ID.") 
        
uid = get_valid_user_id() 
print("Valid user ID entered:", uid)

# %% [markdown]
# # Popularity Base Model

# %%
ratings_grp = books_data.groupby('parent_asin').agg({'rating': [np.size, np.sum, np.mean]})

# %%
ratings_grp

# %%
size_filter = ratings_grp[('rating', 'size') ] > 300
mean_filter = ratings_grp[('rating', 'mean') ] >= 4

books_list = ratings_grp[size_filter & mean_filter]

# %%
popular_books = books_list.sort_values(('rating', 'mean') , ascending=False)
popular_books

# %%
def recommend_popular(df, pop_df, uid, n):
    read_books = df.loc[df['user_id'] == uid, 'parent_asin'].values 
    to_read = [asin for asin in pop_df.index if asin not in read_books] 
    
    return to_read[:n]

# %%
recommend_popular(books_data, popular_books, uid, 3)

# %% [markdown]
# # Item based Collaborative Filtering

# %%
pop_books = books_data[books_data['parent_asin'].isin(popular_books.index)]
pop_books

# %%
pop_books.to_csv('popular_books.csv', index=False)

# %%
um = pop_books.pivot_table(index='user_id', columns='parent_asin', values='rating')
um

# %% [markdown]
# ### Build KNN Model using Utility Matrix

# %%
um_imputed = um.fillna(0)
um_trans_imputed = um_imputed.T

# %%
um_trans_imputed

# %%
nn = NearestNeighbors(n_neighbors=4)
nn.fit(um_trans_imputed)

# %%
neighbors = nn.kneighbors(um_trans_imputed, return_distance=False) 
neighbors

# %%
def recommender_system(user,df, um_mat, neighbors, n):

    consumed = df.loc[df['user_id']==user, 'parent_asin'] # book already read by user
    best_items = df.loc[(df['user_id'] == user) & (df['rating'] == 5), 'parent_asin'] # top rated items
    best_list = []

    for item in best_items:
        idx = um_mat.index.get_loc(item)
        nearest = [um_mat.index[i] for i in neighbors[idx,1:] if um_mat.index[i] not in consumed]  
       

        best_list += list(nearest)

    return pd.Series(best_list).value_counts()[:n]

# %%
recommender_system(uid, pop_books, um_trans_imputed, neighbors, 3)

# %% [markdown]
# ### Build KNN model Using Correlation of um

# %%
um_corr = um.corr()
um_corr_imp = um_corr.fillna(0)

# %%
um_corr_imp

# %%
nn_corr = NearestNeighbors(n_neighbors=4)
nn_corr.fit(um_corr_imp)

# %%
neighbors1 = nn_corr.kneighbors(um_corr_imp, return_distance=False) 
neighbors1

# %%
recommender_system(uid, pop_books, um_corr_imp, neighbors1, 3)

# %% [markdown]
# # SVD Model

# %%
um

# %%
um_imputed

# %%
um_means = np.mean(um_imputed, axis=1)
um_means

# %%
um_demeaned = um_imputed - um_means.values.reshape(-1,1)
um_demeaned

# %%
r = np.linalg.matrix_rank(um_demeaned)
r

# %%
svd = TruncatedSVD(n_components=900, random_state=42)
svd.fit(um_demeaned)

# %%
import pickle
with open('svd.pickle', 'wb') as f:
    pickle.dump(svd,f)

# %%
# svd1 = TruncatedSVD(n_components=100, random_state=42)
# svd1.fit(um_demeaned)

# %%
from sklearn.utils.extmath import randomized_svd 
U, sigma, Vt = randomized_svd(um_demeaned.to_numpy(), n_components=900)

# %%
with open('U_sigma_Vt.pickle', 'wb') as f:
    pickle.dump((U,sigma,Vt),f)

# %%
# U1, sigma1, Vt1 = randomized_svd(um_demeaned.to_numpy(), n_components=100)

# %%
U.shape, sigma.shape, Vt.shape

# %%
sigma = np.diag(sigma)
um_repro = U@sigma@Vt
um_repro +=  um_means.values.reshape(-1,1)

# %%
# sigma1 = np.diag(sigma)
# um_repro1 = U@sigma@Vt
# um_repro1 +=  um_means.values.reshape(-1,1)

# %%
um_repro  = pd.DataFrame(um_repro, index=um_imputed.index, columns=um_imputed.columns) 

# %%
# um_repro1  = pd.DataFrame(um_repro, index=um_imputed.index, columns=um_imputed.columns) 

# %%
# Predict books 
def recommend_books_svd(user, df, um, n): 
    consumed = df.loc[df['user_id']==user, 'parent_asin'] 
    user_books = um.loc[user,:]
    user_books = user_books.sort_values(ascending=False)
    user_books = user_books.drop(index=consumed)

    return user_books.index[:n]

# %%
recommend_books_svd(uid, pop_books , um_repro, 3)

# %% [markdown]
# # RMSE and difference between um and svd-reduced matrix
# 

# %%
rmse = root_mean_squared_error(um_imputed.to_numpy(), um_repro)
print(f"RMSE ({rmse})")

# %%
# rmse1 = root_mean_squared_error(um_imputed.to_numpy(), um_repro1)
# print(f"RMSE ({rmse1})")

# %% [markdown]
# Come up with a metric and train-test-split method to determine which value of k (number of singular values) creates the best SVD-reduced model.

# %% [markdown]
# # Metric: Explained Variance Ratio

# %%
from sklearn.model_selection import train_test_split

# %%
# Split the data into 50% training and 50% testing 
train, test = train_test_split(um_demeaned, test_size=0.5, random_state=42)

# %%
# # Define the number of folds 
# n_splits = 3
# kf = KFold(n_splits=n_splits) 


# Range of k values to test 
k_values = range(1, 101)

# %%
# Store the explained variance ratio for each k 
# explained_variance_ratios = np.zeros((n_splits, len(k_values)))
# initializing a 2-dimensional NumPy array (a matrix) filled with zeros.

explained_variance_ratios = []

# %%
batch_size = um_demeaned.shape[0] // 3

# %%
from sklearn.decomposition import IncrementalPCA

# %%
# # Perform K-Fold Cross-Validation 
# for fold_idx, (train_index, test_index) in enumerate(kf.split(um_demeaned)): #give index of curr fold & generates train & test indices for each fold
#     train, test = um_demeaned.iloc[train_index], um_demeaned.iloc[test_index] 
#     for k_idx, k in enumerate(k_values): 
#         svd = TruncatedSVD(n_components=k) 
#         svd.fit(train) 
#         explained_variance_ratios[fold_idx, k_idx] = svd.explained_variance_ratio_.sum()

# %%
# for k in k_values: 
#     ipca = IncrementalPCA(n_components=k, batch_size=batch_size) 
#     ipca.fit(train) 
#     explained_variance_ratios.append(ipca.explained_variance_ratio_.sum())

# %%
for k in k_values: 
        svd = TruncatedSVD(n_components=k) 
        svd.fit(train) 
        explained_variance_ratios.append(svd.explained_variance_ratio_.sum())

# %%
# Calculate the mean explained variance ratio across folds 
mean_explained_variance_ratio = np.mean(explained_variance_ratios, axis=0)

# %%
# Determine the optimal k 
optimal_k = k_values[np.argmax(mean_explained_variance_ratio)] 
print(f'Optimal k: {optimal_k}')

# %%
# Perform SVD with the optimal k 
U, sigma, Vt = randomized_svd(um_demeaned.to_numpy(), n_components=optimal_k) 
sigma = np.diag(sigma) 
um_repro = U @ sigma @ Vt 
um_repro += um_means.values.reshape(-1, 1) 
um_repro = pd.DataFrame(um_repro, index=um_imputed.index, columns=um_imputed.columns)

# %%
# ipca = IncrementalPCA(n_components=optimal_k, batch_size=batch_size) 
# U = ipca.fit_transform(um_demeaned) 
# Vt = ipca.components_ 
# sigma = ipca.singular_values_ 

# sigma = np.diag(sigma) 
# um_repro = U @ sigma @ Vt 
# um_repro += um_means.values.reshape(-1, 1) 
# um_repro = pd.DataFrame(um_repro, index=um_imputed.index,columns=um_imputed.columns)

# %%
# Evaluate model
rmse = root_mean_squared_error(um_imputed.to_numpy(), um_repro)
print(f"RMSE ({rmse})")



