#Z0096

# import standards
import pandas as pd

# import stats tools
from scipy.stats import chi2_contingency, ttest_ind

# import modeling tools
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report


#################### Explore Functions ####################


def ttest_report(data, sample_mask_one, sample_mask_two, target, alpha=0.05):
    '''
    '''
    
    # get tstat and p value from ttest between two samples
    t, p = stats.ttest_ind(data[sample_mask_one][[target]], data[sample_mask_two][[target]])
    # check p value relative to alpha
    if p < alpha:
        status = 'May reject'
    else:
        status = 'Fail to reject'
    # print results and hypothesis statement
    print(f'''
     T-Stat: {t}
    P-Value: {p}
             {status} the null hypotheses that "{target}" is the same between the two samples.
    ''')


def chi_test(cat, target, alpha=0.05):
    '''    
    '''

    # set observed DataFrame with crosstab
    observed = pd.crosstab(cat, target)
    # assign returned values from chi2_contigency
    chi2, p, degf, expected = chi2_contingency(observed)
    # set expected DataFrame from returned array
    expected = pd.DataFrame(expected)
    # set null hypothesis
    null_hyp = f'"{target.name}" is independent of "{cat.name}'
    # print alpha and p-value
    print(f'''
    alpha: {alpha}
    p-value: {p:.1g}''')
    # print if  p-value is less than significance level
    if p < alpha:
        print(f'''
    Due to p-value {p:.1g} being less than significance level {alpha}, \
may reject the null hypothesis that {null_hyp}."
    ''')
    # print if  p-value is greater than significance level
    else:
        print(f'''
    Due to p-value {p:.1g} being more than significance level {alpha}, \
fail to reject the null hypothesis that {null_hyp}."
    ''')


def gridsearch(X_train, y_train, estimator, params, cv=5, scoring=None):
    '''
    '''
    
    # create grid object from passed arguments
    grid = GridSearchCV(estimator, params, cv=cv, scoring=scoring)
    # fit grid to passed X, y data
    grid.fit(X_train, y_train)
    # loop for each set of params passed
    for params, score in zip(grid.cv_results_['params'],
                             grid.cv_results_['mean_test_score']):

        params['score'] = score
    # create top ten DataFrame of performance per scoring
    grid_df = pd.DataFrame(grid.cv_results_['params']).sort_values(by='score', ascending=False).head(10)
               
    return grid_df


def get_rfe_selected(X_train, y_train, estimator, n_features=None):
    '''
    '''
    
    #
    rfe = RFE(estimator, n_features)
    rfe.fit(X_train, y_train)

    feat_rank = rfe.ranking_
    feat_name = X_train.columns.tolist()

    feat_df = pd.DataFrame({'Feature': feat_name, 'Rank': feat_rank})\
                .sort_values('Rank').reset_index(drop=True)
    
    return feat_df.iloc[:20]


def classifier_scores(y_true, y_pred):
    '''
    '''

    # create dictionary from classifiercation_report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    # print key metrics on positive class predicitons
    print(f'''
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|            *** Model  Report ***            |
|            ---------------------            |
|---------------------------------------------|
|                 Accuracy: {report_dict['accuracy']:>8.2%}          |
|                Precision: {report_dict['1']['precision']:>8.2%}          |
|                   Recall: {report_dict['1']['recall']:>8.2%}          |
|            Total Support: {report_dict['macro avg']['support']:>8}          |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
''')
