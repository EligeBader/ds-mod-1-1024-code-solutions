# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# %%
df = pd.read_csv('mnist.csv')
df

# %%
# Separate features and labels 
X = df.drop(columns=['label'])
y = df['label'] 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# %%
X_scaled

# %%
pca = PCA(n_components=0.90)  # show the components that retain 90% of the variance
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# %%
ncomponents = pca.n_components_
ncomponents


# %%
components = pca.components_
components

# %%
explained_variance = pca.explained_variance_ratio_
explained_variance

# %%
plt.figure(figsize=(8, 6)) 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
 
plt.xlabel('First Principal Component') 
plt.ylabel('Second Principal Component') 
plt.title('MNIST Data after PCA') 
plt.colorbar() 

# %% [markdown]
# ### Try to improve

# %%
pca1 = PCA(n_components=0.90)  
pca1.fit(X_scaled)
X_pca1 = pca1.transform(X_scaled)

# %%
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
lr = LogisticRegression()
lr.fit(xtrain, ytrain)

# %%
preds = lr.predict(xtest)

# %%
print(classification_report(ytest, preds))

# %%
xtrain, xtest, ytrain, ytest = train_test_split(X_pca1, y, test_size=0.25, random_state=42)

# %%
lr_pca = LogisticRegression()
lr_pca.fit(xtrain, ytrain)

# %%
preds = lr_pca.predict(xtest)

# %%
print(classification_report(ytest, preds))

# %%
param_grid = { 
    'C': [0.01, 0.1, 1, 10, 100], #This parameter is the inverse of regularization strength. Lower values of C imply stronger regularization (technique used to prevent overfitting) 
    'solver': ['liblinear', 'lbfgs'] # specifies the algorithm to be used for optimization small and large datasets
    } 


grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5) 
grid_search.fit(xtrain, ytrain) 
best_params = grid_search.best_params_

# %%
best_params

# %%
lr_best = LogisticRegression(max_iter=1000, **best_params) 
lr_best.fit(xtrain, ytrain) 

# %%
best_preds = lr_best.predict(xtest)

# %%
print(classification_report(ytest, preds))

# %% [markdown]
# observation: no improvement

# %%
# Variance thresholds to explore 
variance_thresholds = [0.85, 0.90, 0.95] 
results = []


# experiment with different thresholds for variance retention
for variance_threshold in variance_thresholds: 
     pca = PCA(n_components=variance_threshold) 
     pca.fit(X_scaled)
     X_pca = pca.transform(X_scaled)


     xtrain, xtest, ytrain, ytest = train_test_split(X_pca, y, test_size=0.25, random_state=42)

     #Logistic Regression
     lr = LogisticRegression(max_iter=1000) 
     lr.fit(xtrain, ytrain) 
     
     lr_preds = lr.predict(xtest) 
     lr_report = classification_report(ytest, lr_preds, output_dict=True)

     # Random Forest 
     rf = RandomForestClassifier(n_estimators=100, random_state=42) 
     rf.fit(xtrain, ytrain) 
     
     rf_preds = rf.predict(xtest) 
     rf_report = classification_report(ytest, rf_preds, output_dict=True)


     # Store results 
     results.append({ 
          'variance_threshold': variance_threshold, 
          'lr_report': lr_report, 
          'rf_report': rf_report 
          })
     
     for result in results: 
          print(f"Variance Threshold: {result['variance_threshold']}") 
          print("Logistic Regression Classification Report:") 
          print(classification_report(ytest, lr_preds)) 
          print("\nRandom Forest Classification Report:") 
          print(classification_report(ytest, rf_preds)) 
          print("="*60)


