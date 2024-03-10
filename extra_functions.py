import numpy as np
import pandas as pd 

def split_dataset(dataset, c1 : np.ndarray, c2 : np.ndarray, feature1, feature2, class_train_size):
    data = dataset[[feature1, feature2, 'Class']]
    data = data[(dataset['Class'] == c1) | (dataset['Class'] == c2)]
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    shuffled_class_data = data.sample(frac=1, random_state=42)
    for c in [c1 , c2]:
        class_data = data[data['Class'] == c] 
        train_set = pd.concat([train_set, shuffled_class_data.iloc[:class_train_size]])
        test_set = pd.concat([test_set, shuffled_class_data.iloc[class_train_size:]])
    train_set.replace({
        c1 : -1,
        c2 : 1
    },
    inplace=True)
    test_set.replace({
        c1 : -1,
        c2 : 1
    },
    inplace= True)

    return train_set, test_set
