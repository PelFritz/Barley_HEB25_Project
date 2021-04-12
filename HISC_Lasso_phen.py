from pyHSICLasso import HSICLasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVR
import numpy as np
import pandas as pd

data_2017 = pd.read_csv('data.frame.2017.csv', index_col=0, sep=';')

# For plotting
rsquared_values = []
Phenotype = []
stage = []

# For extracting important metabolites
sel_met = []
corresponding_phen = []
corresponding_stage = []
# Predictions
for phen in ['Bgh0608', 'Bgh0617']:
    print('Phenotype: {}'.format(phen))
    for timepoint in data_2017.groupby('Stage'):
        print('Time point: {}'.format(timepoint[0]))
        metabolites = timepoint[1].drop(columns=['Stage', 'Genotype', 'ID', 'SHO', 'HEA', 'SEL', 'MAT', 'RIP', 'HEI',
                                                 'LOD', 'GEA', 'TGW', 'GRL', 'GRW', 'GRA', 'B', 'Mo', 'P', 'Ca',
                                                 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'Na', 'Mg', 'S', 'K', 'Bgh0519/24',
                                                 'Bgh0608', 'Bgh0617', 'Ph0617'])

        phenotypes = timepoint[1][[phen]].values.ravel()

        # Feature selection with Hilbert Schmidt Independence criterion Lasso
        hsic_lasso = HSICLasso()
        hsic_lasso.input(metabolites.values, phenotypes, featname=metabolites.columns.values)
        hsic_lasso.regression(num_feat=metabolites.shape[1], M=1, B=0, n_jobs=-1)  # use M=30 only for block HSIC lasso
        for metab in hsic_lasso.get_features():
            sel_met.append(metab)
            corresponding_phen.append(phen)
            corresponding_stage.append(timepoint[0])

        # Prediction part
        met_selected = metabolites[hsic_lasso.get_features()]
        scaler = StandardScaler()
        metabolites_stand = scaler.fit_transform(met_selected)
        print(metabolites_stand.shape)

        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

        svr = SVR(kernel='rbf', C=10.0)
        scores = cross_val_score(svr, metabolites_stand, phenotypes, cv=cv, scoring='r2')
        print('score = {}'.format(np.mean(scores)))

        for score in scores:
            rsquared_values.append(score)
            Phenotype.append(phen)
            stage.append(timepoint[0])

df = pd.DataFrame(data={'Phenotype': Phenotype,
                        'Stage': stage,
                        'R-square': rsquared_values,
                        'Model': ['HSIC_SVR']*len(Phenotype)})

df2 = pd.DataFrame(data={'Metabolite': sel_met,
                         'Phenotype': corresponding_phen,
                         'Stage': corresponding_stage})

df2.to_csv('Selected_metabolites_nonlinear.csv', header=True, index=False)
df.to_csv('HSIC_lasso_scores.csv', header=True, index=False)
