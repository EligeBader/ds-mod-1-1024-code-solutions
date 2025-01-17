# %%
import pandas as pd
import numpy as np
import json

# %%
df = pd.read_json('Beauty.json', lines=True)
df

# %% [markdown]
# Write a function that recommends 5 beauty products for each user based on popularity among other users.

# %%
ratings_grp = df.groupby('asin').agg({'overall': [np.size, np.sum, np.mean]})
ratings_grp

# %%
ratings_grp[ratings_grp[('overall', 'size')] > 40]

# %%
popular_prod = ratings_grp[ratings_grp[('overall', 'size')] > 30].sort_values(('overall', 'mean'), ascending=False)
popular_prod

# %%
def recommend_popular(df, popular, id, n): 
    ordered = df.loc[df['reviewerID'] == id, 'asin'].values 
    prod_to_order = [asin for asin in popular.index if asin not in ordered] 
    return prod_to_order[:n] 

# %%
recommend_popular(df, popular_prod, 'A3CIUOJXQ5VDQ2', 5)

# %%
recommend_popular(df, popular_prod, 'ANUDL8U5MQSPX', 5)

# %% [markdown]
# Write a function that recommends 5 beauty products for each user based on next items purchased by other users.

# %%
df

# %%
df_sorted = df.sort_values(['reviewerID', 'unixReviewTime'])

# %%
df_sorted.reset_index(drop=True, inplace= True) 

# %%
df_sorted['next'] = np.nan
df_sorted

# %%
for x in range(len(df_sorted)-1):
    if df_sorted['reviewerID'][x] == df_sorted['reviewerID'][x+1]: 
        df_sorted['next'][x] = df_sorted['asin'][x+1]
    else:
        df_sorted['next'][x] = np.nan

# %%
def recommend_next(id, df, n):
    last_prod = df.loc[df.reviewerID == id, 'asin'].iloc[-1]

    # get last product from all the products for that specific user 
    print(f"last product: {last_prod}")

    # find all the frequency counts to know which product is most ordered after the last_prod
    next_prod = df.loc[df['asin'] == last_prod, 'next'].value_counts(dropna=True)[:n]
    print(F"Next product recommendations: {next_prod}")

# %%
recommend_next('A105A034ZG9EHO', df_sorted, 5)

# %%
recommend_next('AZRD4IZU6TBFV', df_sorted, 5)


