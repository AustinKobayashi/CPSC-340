import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from scipy.sparse import csr_matrix as sparse_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
# YOUR CODE AND REPORT HERE, IN A SENSIBLE FORMAT
filename = "default of credit card clients.xls"

with open(os.path.join("..", "data", filename), "rb") as f:
	credit = pd.read_excel(f,names=("LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "default payment next month"))
credit.head()

X = credit.values[1:,0:23]
Y = credit.values[1:,23:24]

age_of_defaulters = X[np.squeeze(Y==1),4]
print(age_of_defaulters.transpose())
plt.hist(age_of_defaulters.transpose());
plt.show()