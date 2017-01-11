import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

dig_df = pd.read_csv('../Data/train.csv')
X_train = dig_df.drop(['label'], axis=1).astype(np.float)
Y_train = dig_df['label'].astype(np.float)

# Get model
nn = joblib.load('nn.pkl')

# Read in data and get it ready
test_df = pd.read_csv('../Data/test.csv')
X_test = test_df.astype(np.float)
X_test = np.multiply(X_test,(1.0/255.0))

# Run model
Y_predict =  nn.predict(X_test)

Y_predict = np.transpose(Y_predict)


# Make the label column for submission (stupid)
indices = np.arange(1,Y_predict.shape[0] + 1)
index_df = pd.Series(indices)

#Y_predict = np.column_stack((indices,Y_predict))

# Output for competition
submission = pd.DataFrame({'ImageId': indices, 'label':Y_predict})
submission.to_csv('nn_digit_recognizer.csv',float_format = '%d',index=False)
