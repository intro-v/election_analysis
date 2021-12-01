import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math

metric_list = ['total_voters', 'elect_turnout', 'share_of_er', 'share_of_sr', 'share_of_ldpr', 'share_of_com']

def fun_corr_matrix_w_scatters(df_init, color_corr_natrix_flg = True, color_column = None):
    df = df_init
    corr_m = df.corr()
    if color_corr_natrix_flg == True:
        df['tech_column'] = '1'
        color_column = 'tech_column'
    
    iter_list = list(df[color_column].drop_duplicates())
    iter_cnt = len(iter_list)
    c_pal = sns.color_palette("Spectral", 20)
    fig, ax = plt.subplots(6,6)
    
    for k in range(iter_cnt):
        df_tmp = df.loc[df[color_column] == iter_list[k]]
        for i in range(6):
            for j in range(6):
                i_metric = metric_list[i]
                j_metric = metric_list[j]
                if color_corr_natrix_flg == True:
                    color = c_pal[int(- corr_m[i_metric][j_metric]*10 + 10)]
                else:
                    color = c_pal[int(20 * k/iter_cnt)]
                if i!=j:
                    ax[i,j].scatter(df_tmp[j_metric], df_tmp[i_metric], 
                                    color = color,
                                    s = 3,
                                    alpha = 0.6,
                                    label = f"""corr = {corr_m.iloc[i,j]:0.2f}"""
                                    )
                    ax[i,j].legend()
                else:
                    ax[i,j].hist(df_tmp[i_metric],
                                 color = color,
                                 alpha = 0.5
                              )
                ax[i,j].tick_params(axis='both', which='major', labelsize=8)
                if i == 0:
                    ax[i,j].set_title(j_metric)
                if j == 0:
                    ax[i,j].set_ylabel(i_metric)
                if i == 5:
                    ax[i,j].set_xlabel(j_metric)
                if i in [2,3,4,5] and i != j:
                    ax[i,j].set_ylim([0.0, 1.0])
                if j in [2,3,4,5]:
                    ax[i,j].set_xlim([0.0, 1.0])
    if color_corr_natrix_flg == False:
        ax[0,0].legend(iter_list)
    fig.set_size_inches(15, 12)
    fig.show()

def fun_corr_matrix_w_corr_c(df):
    corr_m = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    #mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_m, annot=True, cmap=cmap)
    fig.set_size_inches(12, 8)
    fig.show()
    
def fun_draw_six_gr(df, column, value_list, suptitle = None):
    fig, ax = plt.subplots(3,2)

    for value in value_list:
        mask = (df[column] == value)
        df_tmp = df.loc[mask]
        for i in range(6):
            metric = metric_list[i]
            x = i//2
            y = i%2
            ax[x,y].hist(df_tmp[metric], alpha = 0.4)
            ax[x,y].set_title(metric)
            if x == 0 and y == 0:
                ax[x,y].legend(value_list)

    plt.subplots_adjust(hspace = 0.4)
    fig.set_size_inches(15, 8)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.show()
    