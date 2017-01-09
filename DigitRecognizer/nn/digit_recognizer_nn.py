import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


#..................................................................#
#### Reading and Processing data
dig_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')

# ..................................................................#
# Machine learning
from sklearn.neural_network import MLPClassifier
X_train = dig_df.drop(['label'], axis=1).astype(np.float)
Y_train = dig_df['label'].astype(np.float)
X_test = test_df.astype(np.float)

# Normalize the stuff
X_train = np.multiply(X_train,(1.0/255.0))
X_test = np.multiply(X_test,(1.0/255.0))

# Train
nn = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                   hidden_layer_sizes = (200,), random_state=1)
print('\nTraining Now\n')
nn.fit(X_train,Y_train)
print(nn.score(X_train, Y_train))

# Predicting

Y_predict =  nn.predict(X_test)

# Output for competition
submission = pd.DataFrame({'label':Y_predict})
submission.to_csv('nn_digit_recognizer.csv',index=False)
