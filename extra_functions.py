import numpy as np
import pandas as pd 


def split_dataset(dataset, c1 : np.ndarray, c2 : np.ndarray, feature1, feature2, class_train_size):
    data = dataset[[feature1, feature2, 'Class']]
    data = data[(dataset['Class'] == c1) | (dataset['Class'] == c2)]
    
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    # adding an amount of [class_train_size] from each class to the train set, and an amount of [50 - class_train_size] to the test set
    for c in [c1 , c2]:
        class_data = data[data['Class'] == c] 
        train_set = pd.concat([train_set, class_data.iloc[:class_train_size]])
        test_set = pd.concat([test_set, class_data.iloc[class_train_size:]])

    # encoding
    train_set.replace({c1 : -1,c2 : 1},inplace=True)
    test_set.replace({c1 : -1,c2 : 1},inplace= True)
    
    #shuffling
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_set.drop(['Class'], axis=1).values,  test_set.drop(['Class'], axis=1).values, train_set['Class'].values.reshape(60, 1)  , test_set['Class'].values.reshape(40, 1)
    return X_train, X_test, y_train, y_test
