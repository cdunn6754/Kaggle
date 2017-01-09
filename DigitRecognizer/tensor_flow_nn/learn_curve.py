import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')

#### Reading and Processing data
tit_df = pd.read_csv('../Data/train.csv')

tit_df = tit_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
tit_df = tit_df.replace(['male','female'],[1,0])

# Deal with some NaN
tit_df['Age'][np.isnan(tit_df['Age'])] = tit_df["Age"].mean()

# Deal with missing embarked
def clean_embarked(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    df = df.replace(['S','C','Q'],[0,1,2])
    return df

tit_df = clean_embarked(tit_df)

# ..................................................................#
# Machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Test/Train split
X_train, X_test = train_test_split(tit_df, test_size=0.3)
Y_train_total = X_train['Survived']
X_train_total = X_train.drop(['Survived'], axis=1)
Y_test= X_test['Survived']
X_test= X_test.drop(['Survived'], axis =1)
test_size = float(X_test.shape[0])

# Set up for loop
sizes = np.arange(1,X_train.shape[0],20)
train_output = np.zeros(len(sizes))
test_output = np.zeros(len(sizes))

for i in range(len(sizes)):
    size = sizes[i]
    print('Currently working on size %s' %(str(size)))
    print('\n####################################################')
    
    X_train_ = X_train.sample(size)
    Y_train_ = X_train_['Survived']
    X_train_ = X_train_.drop(['Survived'], axis =1)

    # Train
    nn = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                       hidden_layer_sizes = (20,20), random_state=1)
    nn.fit(X_train_,Y_train_)
    train_output[i] = sum(np.power((nn.predict(X_train_) - Y_train_),2))/float(size)
    test_output[i] = sum(np.power((nn.predict(X_test) - Y_test),2)) / float(test_size)
print np
plt.plot(train_output, label = 'Training')
plt.plot(test_output, label = 'Test')
plt.legend()
plt.show()
