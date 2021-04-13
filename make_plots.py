import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

