import pandas as pd

# candidate metabolites for nutrients prediction
candidate_met = pd.read_csv('HSIC_nut_metab_candidates.csv')
data_2017 = pd.read_csv('data.frame.2017.csv', sep=';')

for df in candidate_met.groupby('Nutrient'):
    print(df[0])
    size = df[1].groupby('Metabolite', as_index=False).size()
    dfs = []
    for df2 in df[1].groupby('Stage'):
        feat = df2[1]['Metabolite'].tolist()
        filter_ = data_2017[data_2017['Stage'] == df2[0]]
        filter_data = pd.concat([filter_[feat], filter_[df[0]]], axis=1)
        corr = filter_data.corr()[df[0]]
        corr.drop(labels=[df[0]], axis=0, inplace=True)
        corr = corr.reset_index().rename(columns={'index': 'Metabolite', df[0]: 'Pearson_corr'})
        corr['Stage'] = [df[0] for i in range(0, corr.shape[0])]
        corr = corr.merge(size, on=['Metabolite']).rename(columns={'size': 'Occurence'})
        dfs.append(corr)
        print(corr.head())

    corr_df = pd.concat(dfs)
    save_dir = 'Candidate_met_' + df[0] + '.csv'
    corr_df.to_csv(save_dir)
