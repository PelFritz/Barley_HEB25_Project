import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import metabolomics data that has been normalized to standards using CRMN package in R
data = pd.read_csv('data.frame.2017.csv', sep=';', index_col='Sample')
stages = data['Stage']
metabolites = data.drop(columns=['Stage', 'Genotype', 'ID', 'SHO', 'HEA', 'SEL', 'MAT', 'RIP', 'HEI', 'LOD',
                                 'GEA', 'TGW', 'GRL', 'GRW', 'GRA', 'B', 'Mo', 'P', 'Ca',
                                 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'Na', 'Mg', 'S', 'K', 'Bgh0519/24',
                                 'Bgh0608', 'Bgh0617', 'Ph0617'])

# Standardizing data
scaler = StandardScaler()
standardized_met = scaler.fit_transform(metabolites)

# Performing PCA to view the structure of the data
pca = PCA(n_components=0.95)
PCs = pca.fit_transform(standardized_met)
Perc_var = pca.explained_variance_ratio_ * 100
PC_col = ['PC{}({}%)'.format(x+1, round(var, 2)) for x, var in enumerate(Perc_var)]
PC_df = pd.DataFrame(data=PCs, columns=PC_col)
PC_df['stage'] = stages.values
print(PCs.shape)
# Visualization
sns.set_theme()
sns.scatterplot(x='PC1(17.88%)', y='PC2(9.49%)', data=PC_df, hue='stage')
plt.show()

# Using the loading scores to get metabolites that contribute the most in PC2
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(data=loadings, columns=['PC{}'.format(x+1) for x in range(0, loadings.shape[1])],
                           index=metabolites.columns)
loadings_PC2 = loadings_df[['PC2']].abs().sort_values(ascending=False, by='PC2')
sel_met = loadings_PC2[loadings_PC2['PC2'] > 0.60]  # selecting metabolites with abs(correlation) > 0.6

