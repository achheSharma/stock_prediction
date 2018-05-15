import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path = ".\\Dataset\\"

filename = "Company.NS_Features"
df = pd.read_csv(path + filename + ".csv")

features = df.columns[1:-1]
x = df.loc[:, features].values
y = df.loc[:,['Target']].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)

eigenvalues = pca.explained_variance_
print(eigenvalues)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])

finalDf = pd.concat([principalDf, df[['Target']]], axis = 1)

pd.DataFrame.to_csv(finalDf,"PCA.csv", index=False)