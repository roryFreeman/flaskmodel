import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
y_name = "class"
print(dataframe.head())
y = dataframe[y_name]
X = dataframe.loc[:, dataframe.columns != y_name]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

filename = 'finalized_model.pkl'

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))

def score(X_test, Y_test):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    return result

def predict(X_test):
    loaded_model = pickle.load(open(filename, 'rb'))
    predicted = loaded_model.predict(X_test)
    return predicted

# train
#train_model(X, y)

# test predict
result = predict(X_test)
print(f'the result is {result}')

print(X_test)






