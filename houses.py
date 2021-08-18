# Code you have previously used to load data
import pandas as pd
from sklearn.model_selection import train_test_split

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# DEALING WITH MISSING VALUES - DROPPING ROWS, IMPUTATION & ...
# DROP - Fill in the lines below: drop columns in training and validation data
    cols_missing_vals = [col for col in X_train.columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_missing_vals, axis=1)
    reduced_X_valid = X_valid.drop(cols_missing_vals, axis=1)



# DEALING WITH CATEGORICAL DATA - DROPPING ROWS, ORDINAL ENC & ONE HOT ENC
# Ordinal
    # Categorical columns in the training data
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if 
                    set(X_valid[col]).issubset(set(X_train[col]))]
            
    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))
            
    print('Categorical columns that will be ordinal encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

# One-Hot 
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Opening file in VS Code from terminal
# Type in CLI: 
#     $ open -a "Visual Studio Code" houses.py

# Words from yesterday and Today's reading
# Kaggle competition reviews and Scholarden