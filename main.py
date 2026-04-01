import pandas as pd
path = ’./Data/train.csv’
data = pd.read_csv(path).to_numpy()
X = data[:,0:-1] #column 1~7 are features
y = data[:,-1] #column 8 is the label
