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


def graph_histogram(df, features):

    #features = ['NB_VISITES','NB_PAGES_VUES_PAR_VISITE','DUREE_SUR_SITEWEB','SCORE_ACTIVITE','SCORE_PROFIL']

    nb_features = len(features)
    nb_graphs = 3 

    fig, axes = plt.subplots(nb_features, nb_graphs)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.set_figheight(25)
    fig.set_figwidth(15)

    i = 0

    for feature in features :
        j=0

        if feature=='DUREE_SUR_SITEWEB' :
            sns.histplot(df[feature],stat='count', ax=axes[i][j])
        else : 
            sns.histplot(df[feature],stat='count', ax=axes[i][j], discrete=True)
        axes[i][j].set_title('Distribution of {}'.format(feature))
        xmax = df[feature].max()
        xmin = df[feature].min()
        axes[i][j].set_xlim(left=xmin, right=xmax)
        axes[i][j].set_xlabel("")
    
        sns.violinplot(y=feature, data=df, ax=axes[i][j+1])
        axes[i][j+1].set_title('Distribution of {}'.format(feature))
        axes[i][j+1].set_ylabel("")

        sns.boxplot(data=df,y=feature, orient="v",ax=axes[i][j+2])
        axes[i][j+2].set_title('Distribution of {}'.format(feature))
        axes[i][j+2].set_ylabel("")

        i=i+1

def graph_histogram_tronque(df, features):

    nb_features = len(features)
    nb_graphs = 3 

    fig, axes = plt.subplots(nb_features, nb_graphs)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.set_figheight(25)
    fig.set_figwidth(15)

    i = 0

    for feature in features :
        j=0

        if feature in ['DUREE_SUR_SITEWEB','DUREE_MOY_PAR_VISITE'] :
            sns.histplot(df[feature],stat='count', ax=axes[i][j])
        else : 
            sns.histplot(df[feature],stat='count', ax=axes[i][j], discrete=True)
        axes[i][j].set_title('Distribution of {}'.format(feature))

        # on coupe les valeurs très élevées pour voire quelques chose sur le graphique
        if feature in ['NB_VISITES','NB_PAGES_VUES_PAR_VISITE','DUREE_SUR_SITEWEB']: 
            xmax = df[feature].quantile(0.995)
        else: 
            xmax = df[feature].max()

        xmin = df[feature].min()
        axes[i][j].set_xlim(left=xmin, right=xmax)
        axes[i][j].set_xlabel("")
    
        sns.violinplot(y=feature, data=df, ax=axes[i][j+1])
        axes[i][j+1].set_title('Distribution of {}'.format(feature))
        axes[i][j+1].set_ylabel("")

        sns.boxplot(data=df,y=feature, orient="v",ax=axes[i][j+2])
        axes[i][j+2].set_title('Distribution of {}'.format(feature))
        axes[i][j+2].set_ylabel("")

        i=i+1




def empirical_distribution(df):

    features = ['NB_VISITES', 'NB_PAGES_VUES_PAR_VISITE', 'DUREE_SUR_SITEWEB','DUREE_MOY_PAR_VISITE']

    nb_features = len(features)
    if np.mod(nb_features,2)==0 :
        nb_rows = int(nb_features/2)
    else :
        nb_rows = int(np.ceil(nb_features/2) ) 

    fig, ax = plt.subplots(nb_rows, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.set_figheight(8)
    fig.set_figwidth(12)

    i = 0
    j = 0
    for feature in features : 
        
        ecdf = ECDF(df[feature])
        x = np.linspace(min(df[feature]), max(df[feature]),num=int(max(df[feature])-min(df[feature])+1) )
        y = ecdf(x)
        sns.lineplot(x, y,drawstyle='steps-pre', ax = ax[i][j])
        ax[i][j].set_title("Distribution empirique {}".format(feature))
        
        j=j+1
        if np.mod(j,2)==0 :
            j = 0
            i=i+1



def graph_distribution_category(df,features):

    nb_features = len(features)
    nb_graphs = 3

    fig, axes = plt.subplots(nb_features, nb_graphs)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.set_figheight(25)
    fig.set_figwidth(16)

    i = 0
    for feature in features :

        j=0
        if feature=='DUREE_SUR_SITEWEB' :
            sns.histplot(data=df, x=feature, hue='CONVERTI', ax=axes[i][j])
        else : 
            sns.histplot(data=df, x=feature, discrete=True, hue='CONVERTI', ax=axes[i][j])
    
        axes[i][j].set_title('Distribution of {}'.format(feature))
        axes[i][j].set_xlabel("")
    
        if feature in ['NB_VISITES', 'NB_PAGES_VUES_PAR_VISITE','DUREE_SUR_SITEWEB']:
        # on coupe la queue de distribution pour voir la forme de la dustribution
            xmax=df[feature].quantile(0.995)
        else :
            xmax=df[feature].max()

        xmin = df[feature].min()
        axes[i][j].set_xlim(left=xmin, right=xmax)

        sns.violinplot(y=feature, x='CONVERTI', data=df, ax=axes[i][j+1])
        axes[i][j+1].set_ylabel("")

        axes[i][j+1].set_title('Distribution of {}'.format(feature))  

        sns.boxplot(data=df,y=feature,x='CONVERTI', orient="v",ax=axes[i][j+2])
        axes[i][j+2].set_title('Distribution of {}'.format(feature))
        axes[i][j+2].set_ylabel("")

        i=i+1


    def plot_matrix_corr(df):
    
        features = ['NB_VISITES', 'DUREE_SUR_SITEWEB','DUREE_MOY_PAR_VISITE', 'SCORE_ACTIVITE','SCORE_PROFIL']

        temp = data[features].copy()
        pd.plotting.scatter_matrix(temp, diagonal='kde',alpha=0.2,  figsize=(12,12))

        fig, ax = plt.subplots(figsize=(8,8)) 
        sns.heatmap(temp.corr(),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
        ax.set_title("Coorélation de Pearson")

        fig, ax = plt.subplots(figsize=(8,8)) 
        #sns.heatmap(data[temp.corr(method='spearman'),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
        ax.set_title("Coorélation de Spearman")

        fig, ax = plt.subplots(figsize=(8,8)) 
        #sns.heatmap(temp.corr(method='kendall'),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
        ax.set_title("Coorélation de Kendall")


def plot_dist_transformation(df):

    features = ['NB_VISITES', 'DUREE_SUR_SITEWEB','DUREE_MOY_PAR_VISITE']
    temp_log = df[features].copy()

    for feature in features :
        temp_log[feature] = np.log(temp_log[feature]+1)

    temp_win = df[features].copy()

    for feature in features :

        quantiles = temp_win[feature].quantile([0.001,0.995]) 
        LOWERBOUND = quantiles[0.001]
        UPPERBOUND = quantiles[0.995]
        temp_win[feature] = np.clip(temp_win[feature], UPPERBOUND, LOWERBOUND)


    nb_features = len(features)
    nb_graphs = 3

    fig, axes = plt.subplots(nb_features*3, nb_graphs)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.set_figheight(35)
    fig.set_figwidth(16)

    i = 0
    for feature in features :
        j=0

        if feature=='NB_VISITES' :
            sns.histplot(df[feature],stat='count', ax=axes[i][j], discrete =True)
        else : 
            sns.histplot(df[feature],stat='count', ax=axes[i][j])

        axes[i][j].set_title('Distribution of {}'.format(feature))
        xmax = df[feature].max()
        xmin = df[feature].min()
        axes[i][j].set_xlim(left=xmin, right=xmax)
        axes[i][j].set_xlabel("")
        
        sns.violinplot(y=feature, data=df, ax=axes[i][j+1])
        axes[i][j+1].set_title('Distribution of {}'.format(feature))
        axes[i][j+1].set_ylabel("")

        sns.boxplot(data=df,y=feature, orient="v",ax=axes[i][j+2])
        axes[i][j+2].set_title('Distribution of {}'.format(feature))
        axes[i][j+2].set_ylabel("")


        j=0
        if feature=='NB_VISITES' :
            sns.histplot(temp_win[feature],stat='count', ax=axes[i+1][j], discrete=True)
        else : 
            sns.histplot(temp_win[feature],stat='count', ax=axes[i+1][j])

        axes[i+1][j].set_title('Dist winsorisation({}+1)'.format(feature))
        xmax = temp_win[feature].max()
        xmin = temp_win[feature].min()
        axes[i+1][j].set_xlim(left=xmin, right=xmax)
        axes[i+1][j].set_xlabel("")
        
        sns.violinplot(y=feature, data=temp_win, ax=axes[i+1][j+1])
        axes[i+1][j+1].set_title('Distribution winsorisation({}+1)'.format(feature))
        axes[i+1][j+1].set_ylabel("")

        sns.boxplot(data=temp_win,y=feature, orient="v",ax=axes[i+1][j+2])
        axes[i+1][j+2].set_title('Distribution winsorisation({}+1)'.format(feature))
        axes[i+1][j+2].set_ylabel("")


        j=0
        if feature=='NB_VISITES' :
            sns.histplot(temp_log[feature],stat='count', ax=axes[i+2][j], discrete=True)
        else : 
            sns.histplot(temp_log[feature],stat='count', ax=axes[i+2][j])

        axes[i+2][j].set_title('Distribution of log({}+1)'.format(feature))
        xmax = temp_log[feature].max()
        xmin = temp_log[feature].min()
        axes[i+2][j].set_xlim(left=xmin, right=xmax)
        axes[i+2][j].set_xlabel("")
        
        sns.violinplot(y=feature, data=temp_log, ax=axes[i+2][j+1])
        axes[i+2][j+1].set_title('Distribution of log({}+1)'.format(feature))
        axes[i+2][j+1].set_ylabel("")

        sns.boxplot(data=temp_log,y=feature, orient="v",ax=axes[i+2][j+2])
        axes[i+2][j+2].set_title('Distribution of log({}+1)'.format(feature))
        axes[i+2][j+2].set_ylabel("")

        i=i+3



def plot_matrix_correlation(df,features):

    temp= df[features].corr()
    fig, ax = plt.subplots(figsize=(8,8)) 
    sns.heatmap(temp.corr(),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
    ax.set_title("Corrélation de Pearson")

    fig, ax = plt.subplots(figsize=(8,8)) 
    sns.heatmap(temp.corr('spearman'),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
    ax.set_title("Corrélation de Spearman")

    fig, ax = plt.subplots(figsize=(8,8)) 
    sns.heatmap(temp.corr('kendall'),annot=True, fmt = ".0%", cmap = "coolwarm", ax=ax)
    ax.set_title("Corrélation de Kendall")



def plot_hist_categorial(features): 

    nb_features = len(features)

    if np.mod(nb_features,2)==0 :
        nb_rows = int(nb_features/2)
    else :
        nb_rows = int(np.ceil(nb_features/2) ) 

    fig, axes = plt.subplots(nb_rows, 2)

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.set_figheight(65)
    fig.set_figwidth(16)

    i = 0 
    j = 0
    nb = 0

    for feature in features :

        series=df_cat[feature].value_counts()/df_cat[feature].notnull().sum()
        sns.barplot(x=series.index, y=series, ax=axes[i][j])
        axes[i][j].set_title('Distribution of {}'.format(feature))
        axes[i][j].set_yticklabels(['{:.0%}'.format(x) for x in axes[i][j].get_yticks()])
        axes[i][j].tick_params(axis='x', rotation=90)
        
        j = j+1
        
        nb = nb+1
        if np.mod(nb,2)==0 :
            i = i+1
            j = 0
