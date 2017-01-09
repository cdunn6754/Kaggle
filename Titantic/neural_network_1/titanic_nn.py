import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')

#### Reading and Processing data
tit_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')

tit_df = tit_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
tit_df = tit_df.replace(['male','female'],[1,0])

test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.replace(['male','female'],[1,0])

# Deal with some NaN
tit_df['Age'][np.isnan(tit_df['Age'])] = tit_df["Age"].mean()
test_df['Age'][np.isnan(test_df['Age'])] = test_df["Age"].mean()
test_df['Fare'][np.isnan(test_df['Fare'])] = test_df['Fare'].mean()

# Deal with missing embarked
def clean_embarked(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    df = df.replace(['S','C','Q'],[0,1,2])
    return df

tit_df = clean_embarked(tit_df)
test_df= clean_embarked(test_df)

# ..................................................................#
# Machine learning
from sklearn.neural_network import MLPClassifier

X_train = tit_df.drop(["Survived","PassengerId"], axis=1)
Y_train = tit_df['Survived']
X_test = test_df.drop(["PassengerId"], axis=1)

# Train
nn = MLPClassifier(solver='lbfgs', alpha = 14,
                   hidden_layer_sizes = (20,20), random_state=1)
nn.fit(X_train,Y_train)
print( nn.score(X_train, Y_train))

# Predicting
Y_predict =  nn.predict(X_test)

# Output for competition
submission = pd.DataFrame({"PassengerId": test_df['PassengerId'],'Survived':Y_predict})
submission.to_csv('nn_titanic.csv',index=False)
