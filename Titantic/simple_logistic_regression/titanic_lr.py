import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')


# Processing data
tit_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')

passenger_id = test_df['PassengerId']

tit_df = tit_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
tit_df = tit_df.replace(['male','female'],[1,0])

test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.replace(['male','female'],[1,0])

# Machine learning
from sklearn.linear_model import LogisticRegression

X_train = tit_df.drop(["Survived"], axis=1)
Y_train = tit_df['Survived']

X_test = test_df

# Deal with NAN
average_age_tit = tit_df["Age"].mean()
average_age_test = test_df["Age"].mean()

X_train['Age'][np.isnan(tit_df['Age'])] = average_age_tit
X_test['Age'][np.isnan(test_df['Age'])] = average_age_test
X_test['Fare'][np.isnan(test_df['Fare'])] = X_test['Fare'].mean()
logreg = LogisticRegression()

logreg.fit(X_train,Y_train)
print(logreg.score(X_train, Y_train))

Y_predict = logreg.predict(X_test)

submission = pd.DataFrame({"PassengerId": passenger_id,'Survived':Y_predict})
submission.to_csv('titanic.csv',index=False)
