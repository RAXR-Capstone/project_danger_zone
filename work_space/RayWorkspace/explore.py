#Z0096

# import standards
import pandas as pd

# import stats tools
from scipy.stats import chi2_contingency, ttest_ind


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