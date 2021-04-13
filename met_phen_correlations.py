import pandas as pd

pd.options.display.width = 0
final_metabolites = pd.read_csv('Final_selected_metabolites.csv')
cols = final_metabolites['Metabolite'].tolist() + ['Bgh0608', 'Bgh0617', 'Stage']

data = pd.read_csv('data.frame.2017.csv', sep=';', usecols=cols)
data = data[data['Stage'] != 'D']

dfs = []
for df in data.groupby('Stage'):
    corr = df[1].corr(method='spearman')[['Bgh0608', 'Bgh0617']]
    corr.drop(labels=['Bgh0608', 'Bgh0617'], axis=0, inplace=True)
    corr = corr.reset_index().rename(columns={'index': 'Metabolite', 'Bgh0608': 'Bgh0608_corr',
                                              'Bgh0617': 'Bgh0617_corr '})
    corr['Stage'] = [df[0]] * corr.shape[0]
    dfs.append(corr)

dfs_corr = pd.concat(dfs)

df_final = final_metabolites.merge(dfs_corr, on=['Metabolite', 'Stage'])
df_final.drop(columns=['B_graminis_08', 'B_graminis_17'], inplace=True)
df_final.round(2).to_csv('candidate_met.csv', index=False)
