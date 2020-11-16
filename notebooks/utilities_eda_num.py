import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

import src.settings.base as stg
import src.settings.column_names as column_names
import src.infrastructure.make_dataset as make_dataset
import src.domain.cleaning as cleaning



def top_variables_discretes(df,features):

    zoom = df.filter(items=features)

    for feature in features :
        top5 = zoom.sort_values(by=feature, ascending=False).head()
        print('\nTop5 {}\n'.format(feature),top5,"\n")


def zoom_duree_0(df,features):
    
    zoom = df.filter(items=features).sort_values(by='DUREE_SUR_SITEWEB')
    zoom = zoom.query('NB_VISITES!=0 & DUREE_SUR_SITEWEB==0')
    print(zoom)    

def correct_outliers_errors(df):
        df = df.copy()
    
        mask_outlier = (df[stg.NB_VISITES_COL] != 0) & (df[stg.DUREE_SUR_SITEWEB_COL]==0) 
        df[stg.NB_VISITES_COL] = np.where(mask_outlier, np.nan, df[stg.NB_VISITES_COL])
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where(mask_outlier, np.nan, df[stg.DUREE_SUR_SITEWEB_COL])

        df[stg.NB_VISITES_COL] = np.where((df[stg.NB_VISITES_COL]==251) & (df.index==stg.OUTLIER_ID), np.nan, df[stg.NB_VISITES_COL])
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where((df[stg.DUREE_SUR_SITEWEB_COL]==49.0) & (df.index==stg.OUTLIER_ID), np.nan, df[stg.DUREE_SUR_SITEWEB_COL])
    
        return df    


def add_duree_moy(df):

    df = df.copy()
    df['DUREE_MOY_PAR_VISITE']=np.nan
    df['DUREE_MOY_PAR_VISITE'].loc[df['NB_VISITES']>0] = df['DUREE_SUR_SITEWEB']/df['NB_VISITES']
    df['DUREE_MOY_PAR_VISITE'].loc[df['NB_VISITES']==0] = 0
    return df


def display_quantile(df, features) :

    report_quantile = df[features].quantile(q=[0,0.20,0.5,0.95,0.99,0.995,1])
    report_quantile.index = ["{:.1%}".format(q) for q in report_quantile.index]
    print("Quantiles de distribution")
    print(report_quantile)



def test_stat_same_distribution(df,features):

    KS_statistics = []
    KS_pvalues = []
    MWhitney_statistics = []
    MWhitney_pvalues = []
    Kruskal_statistics = []
    Kruskal_pvalues = []

    for feature in features :

        df1 = df.query('CONVERTI==1').filter(items=[feature]).dropna().to_numpy().flatten()
        df0 = df.query('CONVERTI==0').filter(items=[feature]).dropna().to_numpy().flatten()

        KS_statistic, KS_pvalue = stats.ks_2samp(df1,df0)
        # ne marche pas 
        #statistic, pvalue = stats.wilcoxon(df1,df0)
        MWhitney_statistic, MWhitney_pvalue = stats.mannwhitneyu(df1,df0)
        Kruskal_statistic, Kruskal_pvalue = stats.kruskal(df1,df0)

        KS_statistics.append(KS_statistic)
        KS_pvalues.append(KS_pvalue)
        MWhitney_statistics.append(MWhitney_statistic)
        MWhitney_pvalues.append(MWhitney_pvalue)
        Kruskal_statistics.append(Kruskal_statistic)
        Kruskal_pvalues.append(Kruskal_pvalue)

    d1 = {'KS_statistic': KS_statistics, 'KS_pvalue': KS_pvalues} 
    d2= {'MW_statistic': MWhitney_statistics, 'MW_pvalue': MWhitney_pvalues} 
    d3= {'Kruskal_statistic': Kruskal_statistics, 'Kruskal_pvalue': Kruskal_pvalues}

    report1 = pd.concat([pd.Series(v, name=k) for k, v in d1.items()], axis=1)
    report2 = pd.concat([pd.Series(v, name=k) for k, v in d2.items()], axis=1)
    report3 = pd.concat([pd.Series(v, name=k) for k, v in d3.items()], axis=1)

    report1['KS_pvalue>0.05'] =  report1['KS_pvalue']>0.05
    report2['MW_pvalue>0.05'] =  report2['MW_pvalue']>0.05
    report3['Kruskal_pvalue>0.05'] =  report3['Kruskal_pvalue']>0.05
    
    report1.index = features
    report2.index = features
    report3.index = features

    #print(report1,'\n')
    print(report2,'\n')
    print(report3,'\n')

