import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
pd.options.display.width = 0
seed = 42
perm_cv = 10
data_2017 = pd.read_csv('data.frame.2017.csv', index_col=0, sep=';')
print(data_2017.head())
lasso, ridge, Enet = Lasso(max_iter=100000), Ridge(max_iter=100000), ElasticNet(max_iter=100000)

# For plotting
stage = []
model_type = []
avg_score = []
phenotype_name = []

# For extracting important metabolites
predictive_features_df = []

for timepoint in data_2017.groupby('Stage'):
    print('Time point: {}'.format(timepoint[0]))
    metabolites = timepoint[1].drop(columns=['Stage', 'Genotype', 'ID', 'SHO', 'HEA', 'SEL', 'MAT', 'RIP', 'HEI', 'LOD',
                                             'GEA', 'TGW', 'GRL', 'GRW', 'GRA', 'B', 'Mo', 'P', 'Ca',
                                             'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'Na', 'Mg', 'S', 'K', 'Bgh0519/24',
                                             'Bgh0608', 'Bgh0617', 'Ph0617'])
    print(metabolites.shape)
    phenotypes = timepoint[1][['Bgh0608', 'Bgh0617']]

    scaler = StandardScaler()
    metabolites_stand = scaler.fit_transform(metabolites)

    for regressor, reg_name in zip([lasso, ridge, Enet], ['Lasso', 'Ridge', 'E_net']):
        print('Regressor: {}'.format(reg_name))
        for phen in phenotypes.columns:
            y = phenotypes[[phen]]
            parameters = {'alpha': np.arange(0, 1, 0.01)[1:]}
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)

            grid_searcher = GridSearchCV(regressor, parameters, cv=cv, scoring='r2')
            grid_searcher.fit(metabolites_stand, y)
            best_model = grid_searcher.best_estimator_
            score = grid_searcher.best_score_ if grid_searcher.best_score_ > 0 else 0.0
            alpha = grid_searcher.best_params_['alpha']
            print('Phenotype: {}, score: {}, alpha: {}'.format(phen, score, alpha))

            cv_score = cross_val_score(best_model, metabolites_stand, y, cv=cv, scoring='r2')
            for sc in cv_score:
                stage.append(timepoint[0])
                phenotype_name.append(phen)
                model_type.append(reg_name)
                avg_score.append(sc)

            # Obtaining predictive metabolites using permutation based method
            permutated_r2 = permutation_importance(best_model, metabolites_stand, y, scoring='r2',
                                                   n_repeats=10, random_state=42)
            mean_importance = permutated_r2.importances_mean
            Std_importance = permutated_r2.importances_std

            feature_imp = pd.DataFrame(data={'Metabolite': metabolites.columns,
                                             'mean_importance': mean_importance,
                                             'std_importance': Std_importance,
                                             'Stage': [timepoint[0]] * len(mean_importance),
                                             'phenotype': [phen] * len(mean_importance),
                                             'Model': [reg_name] * len(mean_importance)})
            feature_imp.sort_values(by='mean_importance', inplace=True, ascending=False)
            feature_imp = feature_imp.head(100)
            predictive_features_df.append(feature_imp)
            print(feature_imp.head())
            print(feature_imp.shape)

df = pd.DataFrame({
    'Phenotype': phenotype_name,
    'Stage': stage,
    'R-square': avg_score,
    'Model': model_type})
df.to_csv('linear_model_scores.csv', index=False)

predictive_features = pd.concat(predictive_features_df)
predictive_features.to_csv('Predictive_features.csv', index=False, header=True)
