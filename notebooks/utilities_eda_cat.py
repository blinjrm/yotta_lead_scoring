
import pandas as pd
import numpy as np


def report_cat(data_cat) :

    nb_null=data_cat.isnull().sum()
    null_freq_pour=data_cat.isnull().sum()/len(data_cat)*100

    nb_category = []
    for col in data_cat.columns :
        nb_category.append(data_cat[col].nunique())
    nb_category = pd.Series(nb_category)
    nb_category.index = data_cat.columns

    df_null=pd.concat([nb_null,null_freq_pour, nb_category], axis=1) 
    df_null.columns=['MISSING','MISSING %', 'DISTINCT']
    print("number of missing values",'\n',df_null.sort_values(by=['MISSING %'], ascending=False))



from scipy.stats import chi2_contingency

def test_independance_khi2 (data_cat,y, alpha):

    def test_ki2(x, y, alpha) :

        confusion_matrix = pd.crosstab(x,y)
        stat, pvalue, degre, array = chi2_contingency(confusion_matrix)

        if (pvalue > alpha) :
            decision = "Ho - independant"
        else:
            decision =  "H1 - dependant"
        return (stat, pvalue, decision)

    res = []
    for col in data_cat.columns :
        res_test = test_ki2(data_cat[col], y, alpha)
        res.append(res_test)

    df = pd.DataFrame(res, columns =['STATISTIC', 'P-VALUE', 'DECISION'])
    df.index = data_cat.columns
    df = df.sort_values(by=['P-VALUE'])
    
    print(df)



from scipy.stats import chi2_contingency


def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)
    nb_rows = confusion_matrix.shape[0]
    nb_cols = confusion_matrix.shape[1]
    n = confusion_matrix.sum().sum()

    chi2 = chi2_contingency(confusion_matrix)[0]
    phi = chi2/n

    phi_non_corr = phi/min(nb_rows-1,nb_cols-1)
    cramer = np.sqrt(phi_non_corr)

    phi_corr = max(0,phi - (nb_rows-1)*(nb_cols-1)/(n-1))

    nb_rows_corr = nb_rows - (nb_rows-1)*(nb_rows-1)/(n-1)
    nb_cols_corr = nb_cols - (nb_cols-1)*(nb_cols-1)/(n-1)

    cramer_corr = np.sqrt(phi_corr/((nb_rows_corr-1)*(nb_cols_corr-1)))
        
    return cramer_corr



def matrix_cramer(data_cat) :


    nb_col = data_cat.shape[1]
    Cramer= np.empty([nb_col,nb_col])

    i=0
    for feature1 in data_cat.columns :
        j=0
        for feature2 in data_cat.columns :
            Cramer[i,j]=cramers_v(data_cat[feature1],data_cat[feature2])

            j=j+1
        i=i+1

    cramer = pd.DataFrame(Cramer, columns=data_cat.columns, index = data_cat.columns)
    return cramer