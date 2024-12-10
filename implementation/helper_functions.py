# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, OneHotEncoder, power_transform, PowerTransformer
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import yeojohnson
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import dill
import os

# %%
def load_data(file):
    df = pd.read_csv(file)

    return df

    

with open('read_file.pickle', 'wb') as f:
    dill.dump(load_data, f)

# %%
def split_data(df, target, feature_selected= None, features =[]):
    if feature_selected == None:
        X = df.drop(columns= [target] + features)
        y = df[target]

    else:
        X = df[feature_selected]
        y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



with open('split_data.pickle', 'wb') as f:
    dill.dump(split_data, f)

# %%
def clean_data(df):
    #Use SimpleImputer

    #Check Columns having nulls
    nulls_col = df.columns[df.isnull().sum() > 0]
    nulls_col  = list(nulls_col)


    # Separate numeric and categorical features
    numeric_features = [feat for feat in nulls_col if df[feat].dtype.kind in 'bifc']
    categorical_features = [feat for feat in nulls_col if feat not in numeric_features]

    # Impute missing values for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Impute missing values for categorical features    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])


    return df



with open('clean_data.pickle', 'wb') as f:
    dill.dump(clean_data, f)

# %%
def encode_data(df, target, categorical_cols, train):
    file_name = 'trained_data.pickle'
    if not train: 
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                te = dill.load(f)

             # Transform the categorical columns 
            tempvar = te.transform(df[categorical_cols]) 
    
            # Update the original DataFrame with encoded values 
            for i, col in enumerate(categorical_cols): 
                df[col] = tempvar[:, i]

    else:

        # Initialize and fit the TargetEncoder 
        te = TargetEncoder() 
        te.fit(df[categorical_cols], df[target]) 

        
        # Transform the categorical columns 
        tempvar = te.transform(df[categorical_cols]) 
        
        # Update the original DataFrame with encoded values 
        for i, col in enumerate(categorical_cols): 
            df[col] = tempvar[:, i]
        

        with open(file_name, 'wb') as f:
            dill.dump(te, f)

    return df


with open('encode_data.pickle', 'wb') as f:
    dill.dump(encode_data, f)

# %%
def transform_data(df, target):
    file_name ="powertransformer.pickle"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            pt = dill.load(f)

        pt = PowerTransformer(method='yeo-johnson')
        pt.fit(df[target].values.reshape(-1,1))
    
    else:
        pt = PowerTransformer(method='yeo-johnson')
        pt.fit(df[target].values.reshape(-1,1))

        yj_target = pt.transform(df[target].values.reshape(-1,1))
        df['transform_target'] = yj_target

        with open(file_name, 'wb') as f:
            dill.dump(pt, f)

    return df


with open('transform_data.pickle', 'wb') as f:
    dill.dump(transform_data, f)

# %%
def train_model(model_class, xtrain, ytrain, **args):
    model = model_class(**args)
    model.fit(xtrain, ytrain)

    return model



with open('train_model.pickle', 'wb') as f:
    dill.dump(train_model, f)   

# %%
def predict_model(df, model, features = []):

    file_name = "powertransformer.pickle"

    X_new = df.drop(columns=features)
    y_new_pred = model.predict(X_new)


    print(f"Model's raw prediction: {y_new_pred}")

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
             pt =  pickle.load(f)


        if hasattr(pt, 'inverse_transform'): 
           try: 
               y_new_pred = pt.inverse_transform(y_new_pred.reshape(-1, 1)).flatten() 
               if np.isnan(y_new_pred).any(): 
                    print("Inverse transform produced NaNs. Returning raw predictions.") 
                    y_new_pred = model.predict(X_new) 
           except Exception as e: 
                    print(f"Inverse transform failed: {e}") 
                    y_new_pred = model.predict(X_new)
        else: 
            print("Loaded transformer does not have the inverse_transform method.")

    
    print(f"Predictions after inverse transform (if applicable):{y_new_pred}")

    return y_new_pred


with open('predict_model.pickle', 'wb') as f:
    dill.dump(predict_model, f) 


