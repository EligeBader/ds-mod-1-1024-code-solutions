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
from sklearn.impute import SimpleImputer, KNNImputer
import dill
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

# %%
def load_data(file):
    df = pd.read_csv(file)

    return df

    

with open('read_file.pickle', 'wb') as f:
    dill.dump(load_data, f)

# %%
def drop_features(df, features_to_drop=[]):
    df = df.drop(columns=features_to_drop)

    return df

with open('drop_features.pickle', 'wb') as f:
    dill.dump(drop_features, f)

# %%
def split_data(df, target,col_dropped, feature_selected= None):
    if feature_selected == None:
        X = df.drop(columns= [target] + col_dropped)
        y = df[target]

    else:
        X = df[feature_selected]
        y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



with open('split_data.pickle', 'wb') as f:
    dill.dump(split_data, f)

# %%
def clean_data(df, train=True, target= [], feature_names=None):
    #Use SimpleImputer
    numeric_imputer_file = 'numeric_imputer.pickle' 
    categorical_imputer_file = 'categorical_imputer.pickle'

    
    # Separate numeric and categorical features
    numeric_features = df.drop(columns=target).select_dtypes(exclude=object).columns.tolist()
    categorical_features = df.select_dtypes(include=object).columns.tolist()
    
    

    if train:

        if numeric_features:
            # Impute missing values for numeric features
            numeric_imputer = KNNImputer(n_neighbors = 5)
            numeric_imputer.fit(df[numeric_features])
            df[numeric_features] = numeric_imputer.transform(df[numeric_features])

            with open(numeric_imputer_file, 'wb') as f: 
                dill.dump(numeric_imputer,f)

            # print(numeric_features)

        if categorical_features:
            # Impute missing values for categorical features    
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            categorical_imputer.fit(df[categorical_features])
            df[categorical_features] = categorical_imputer.transform(df[categorical_features])

            feature_names = df.columns.tolist()
            
            with open(categorical_imputer_file, 'wb') as f: 
                dill.dump(categorical_imputer,f)

    else:
        
        if os.path.exists('feature_names.pickle'): 
            with open('feature_names.pickle', 'rb') as f: 
                feature_names = dill.load(f)
                

            for feature in feature_names: 
                if feature not in df.columns: 
                    df[feature] = 0

            df = df[feature_names]

        if numeric_features and os.path.exists(numeric_imputer_file):
            
            with open(numeric_imputer_file, 'rb') as f: 
                numeric_imputer = dill.load(f)
            # print(numeric_features)
            # print(numeric_imputer.get_feature_names_out())
            
            df[numeric_features] = numeric_imputer.transform(df[numeric_features]) 

            # print(numeric_features)

        if categorical_features and os.path.exists(categorical_imputer_file): 

            with open(categorical_imputer_file, 'rb') as f: 
                categorical_imputer = dill.load(f) 

            df[categorical_features] = categorical_imputer.transform(df[categorical_features])


    return df



with open('clean_data.pickle', 'wb') as f:
    dill.dump(clean_data, f)


# %%
def encode_data(df, encoding_methods, train, target=[]): 
    
    file_name = 'encoders.pickle' 
    
    target_cols = [col for col, method in encoding_methods.items() if method == 'target' and col in df.columns] 
    ordinal_cols = [col for col, method in encoding_methods.items() if method == 'ordinal' and col in df.columns] 
    encoders = {} 
   
    if train: 
        
        if target_cols: 
            target_encoder = TargetEncoder() 
            target_encoder.fit(df[target_cols], df[target]) 
            encoded_data = target_encoder.transform(df[target_cols]) 
            for i, col in enumerate(target_cols):
                df[col] = encoded_data[:, i]
            encoders['target'] = target_encoder
          
        if ordinal_cols: 
            ordinal_encoder = OrdinalEncoder() 
            ordinal_encoder.fit(df[ordinal_cols]) 
            df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])
            encoders['ordinal'] = ordinal_encoder
            
        with open(file_name, 'wb') as f: 
            dill.dump(encoders, f) 


    else: 
        if os.path.exists(file_name): 
            with open(file_name, 'rb') as f: 
                encoders = dill.load(f) 

            print(encoders) 
            # Transform the categorical columns 
            if 'target' in encoders and target_cols: 
                encoded_data = encoders['target'].transform(df[target_cols])
                for i, col in enumerate(target_cols):
                    df[col] = encoded_data[:, i]
                

            if 'ordinal' in encoders and ordinal_cols: 
                df[ordinal_cols] = encoders['ordinal'].transform(df[ordinal_cols]) 
        else: 
            raise FileNotFoundError(f"Encoders file not found: {file_name}")
    
    return df


with open("encode_data.pickle", 'wb') as f:
    dill.dump(encode_data,f)



# %%
def transform_data(df, target):
    file_name ="powertransformer.pickle"

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            pt = dill.load(f)

        yj_target = pt.transform(df[target].values.reshape(-1,1))
        df['transform_target'] = yj_target
    

    else:
        pt = PowerTransformer(method='box-cox')
        pt.fit(df[target].values.reshape(-1,1))

        yj_target = pt.transform(df[target].values.reshape(-1,1))
        df['transform_target'] = yj_target

        with open(file_name, 'wb') as f:
            dill.dump(pt, f)

    return df


with open('transformed_data.pickle', 'wb') as f:
    dill.dump(transform_data, f)

# %%
def train_model(model_class, xtrain, ytrain, param_grid={}, best_combination=False ,  **args):
    
    if best_combination == True:

        model = model_class(**args)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(xtrain, xtrain)

        best_model = grid_search.best_estimator_
        best_model.fit(xtrain, xtrain)

        return best_model
    
        with open('trained_model.pickle', 'wb') as f:
            dill.dump(best_model, f)

    else:
        model = model_class(**args)
        model.fit(xtrain, ytrain)

        return model

        with open('trained_model.pickle', 'wb') as f:
            dill.dump(model, f)



with open('train_model.pickle', 'wb') as f:
    dill.dump(train_model, f)   

# %%
def predict_model(df, model):

    file_name = "powertransformer.pickle"
    
    y_new_pred = model.predict(df)

    
    print(f"Model's raw prediction: {y_new_pred}")

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
             pt =  pickle.load(f)


        if hasattr(pt, 'inverse_transform'): 
           try: 
               if not np.any(np.isnan(y_new_pred)) and np.all(np.isfinite(y_new_pred)):
                y_new_pred = pt.inverse_transform(y_new_pred.reshape(-1, 1)).flatten() 

                with open('predicted_model.pickle', 'wb') as f:
                    dill.dump(y_new_pred, f) 

                if np.isnan(y_new_pred).any(): 
                    print("Inverse transform produced NaNs. Returning raw predictions.") 
                    y_new_pred = model.predict(df) 
           except Exception as e: 
                    print(f"Inverse transform failed: {e}") 
                    y_new_pred = model.predict(df)
        else: 
            print("Loaded transformer does not have the inverse_transform method.")

    
    print(f"Predictions after inverse transform (if applicable):{y_new_pred}")

    return y_new_pred


with open('predict_model.pickle', 'wb') as f:
    dill.dump(predict_model, f) 


