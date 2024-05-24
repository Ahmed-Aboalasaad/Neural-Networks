import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\\Users\\p c\\Desktop\\University\\NN\\Labs\\Neural-Networks\\MLP\\data.csv')
X = data.iloc[:, :5]
Y = data.iloc[:, -3:]
X = pd.DataFrame(X)
y = pd.DataFrame(Y)

for one, two in zip(X.iterrows(), Y.iterrows()):
    print(one)
    print(two)
    exit(0)

# X_df = print(type(pd.DataFrame(X)))
# Y_df = print(type(pd.DataFrame(Y)))

# for index, (record, label) in enumerate(zip(X_df.values(), Y_df.values())):
#     print("Record:", record[1].values)
#     print("Label:", label[1].values)
#     if index >= 5:  # Print only first 5 records for demonstration
#         break