import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Comparing Linear models and nonlinear HSIC_SVR for phenotype predictions with metabolites
scores_linear = pd.read_csv('linear_model_scores.csv')
scores_svr = pd.read_csv('HSIC_lasso_scores.csv')
scores = pd.concat([scores_svr, scores_linear])
scores.replace({'A': 'Shooting', 'B': 'Heading', 'C': 'Ears',
                'D': 'Senescence'}, inplace=True)
scores['R-square'] = np.clip(scores['R-square'].values, 0, None)  # Clipping negative R-square scores
print(scores.head())

# plot
order = np.sort(scores['Phenotype'].unique())
g = sns.catplot(x='Stage', y='R-square', hue='Stage', row='Phenotype', col='Model',
                data=scores, legend_out=True, kind='box', margin_titles=True, sharex=True, sharey=True)
g.set_xticklabels(rotation=90)
plt.show()

# Presenting predictive performances on nutrients using metabolites
nut_scores = pd.read_csv('HSIC_nutrients.csv')
nut_scores['Stage'].replace({'A': 'Shooting', 'B': 'Heading', 'C': 'Ears',
                    'D': 'Senescence', 'Phenotype': 'Nutrients'}, inplace=True)
nut_scores.rename(columns={'Phenotype': 'Nutrient'}, inplace=True)
fig, ax = plt.subplots(1, 6, figsize=(16, 5), sharey='row')
for i, df in enumerate(nut_scores.groupby('Nutrient')):
    g = sns.boxplot(x='Stage', y='R-square', hue='Stage', data=df[1], ax=ax[i], showmeans=True,
                    meanprops={"marker": "o",
                               "markerfacecolor": "white",
                               "markeredgecolor": "black",
                               "markersize": "8"})
    sns.stripplot(x='Stage', y='R-square', hue='Stage', data=df[1], alpha=0.6, ax=ax[i])
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[i].legend([], [], frameon=False)
    ax[i].set_title(df[0], fontweight='bold')

plt.show()
