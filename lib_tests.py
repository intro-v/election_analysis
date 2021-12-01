import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy.stats as stats

def fun_draw_distribution_nicely(series_list, name_list, title, bins_based_on = 0):
    """Рисует 2 графика: гистограммама и боксплот
    
    series_list - данные [data_list_1, data_list_2, data_list_3, ...] 
    name_list - название данных, добавлем это в легенду [name_1, name_2, name_3, ...]
    title - заголовок над графиками
    bins_based_on - при построение гистограмм, бины рассчитываются по этому data_list
    """
    fig, ax = plt.subplots(2,1)
    # чтобы бины у распределений были как у заданного
    counts, bins = np.histogram(series_list[bins_based_on])
    for i in range(len(series_list)):
        ax[0].hist(series_list[i], alpha = 0.3, bins = bins, label = name_list[i] + ' ' + str(series_list[i].shape[0])) 
    fig.suptitle(title, fontsize=20)
    ax[0].legend()
    series_list_back = [k for k in reversed(series_list)]
    name_list_back = [k for k in reversed(name_list)]
    ax[1].boxplot(series_list_back, labels = name_list_back, vert = False)
    ax[1].set_xlabel(title)
    fig.set_size_inches(10,6)
    fig.show() 
    
    
def ztest(mean_1, std_1, mean_2, std_2):
    """Считаем ztest для двух выборок по их параметрам"""
    stat_mean = mean_1 - mean_2
    stat_std = np.sqrt( std_1**2 + std_2**2 )
    z_score = stat_mean/stat_std
    p_value = 2*stats.norm().sf(abs(z_score))
    return z_score, p_value

def fun_do_bootstrap(data_list, draw_flg=False): 
    """Бутстреп для среднего: принимаем лист с данными, возвращаем параметры распределения среднего по выборке
    """
    boot_means = []
    for _ in range(10000):
        bootsample = np.random.choice(data_list,size=len(data_list), replace=True)
        boot_means.append(bootsample.mean())
    b_mean = np.mean(boot_means)
    b_std = np.std(boot_means)
    if draw_flg == True:
        fig, ax = plt.subplots()
        ax.hist(boot_means, alpha = 0.5)
        ax.hist(norm(b_mean, b_std).rvs(size = 10000), alpha = 0.3)
        ax.legend(['bootstrap sample', 'norm distribution'])
    return b_mean, b_std, boot_means

def fun_draw_FPR_aa_test(ax_0, ax_1, p_list, title = ''):
    
    ax_0.hist(p_list)
    ax_0.set_title('p-value in aa-test')
    p_list.sort()
    y = 1. * np.arange(len(p_list)) / (len(p_list) - 1)
    ax_1.scatter(x = p_list, y = y)
    ax_1.plot([0,1], [0,1])
    ax_1.set_title('CDF p-value in aa-test')
    ax_0.set_xlabel('p-value')
    ax_1.set_xlabel('p-value')
    ax_1.set_ylabel('FPR')
    
    
def fun_perform_aa_test(ax_0, ax_1, test, df, b_gr_size, i=1000, **kwargs):
    """Проводим аа-тест и рисуем на принимаемых графиках распределение и коммулятивное распределение p-value"""
    if test in ('ttest', 'mannwhitneyu', 'ztest_for_global_share', 'bootstrap'):
        pvalue_list = []

        if test == 'ttest':
            data_list = df[kwargs['metric']]
            for _ in range(i):
                sample_data_list = list(np.random.choice(data_list, b_gr_size))
                T, pvalue = stats.ttest_ind(data_list, sample_data_list, equal_var=False)
                pvalue_list = pvalue_list + [pvalue]
        
        if test == 'mannwhitneyu':
            data_list = df[kwargs['metric']]
            for _ in range(i):
                sample_data_list = list(np.random.choice(data_list, b_gr_size))
                U, pvalue = stats.mannwhitneyu(data_list, sample_data_list, method="exact")
                pvalue_list = pvalue_list + [pvalue]
                
        if test == 'ztest_for_global_share':
            data_list = df.loc[:, [kwargs['num'], kwargs['den']]].values.tolist()
            a_num = sum([i[0] for i in data_list ])
            a_den = sum([i[1] for i in data_list ])
            for _ in range(i):
                sample_data_list = [data_list[i] for i in list(np.random.choice(len(data_list),b_gr_size)) ]
                b_num = sum([i[0] for i in sample_data_list ])
                b_den = sum([i[1] for i in sample_data_list ])
                Z, pvalue = ztest_for_share(a_num, a_den, b_num, b_den)
                pvalue_list = pvalue_list + [pvalue]
        
        if test == 'bootstrap':
            data_list = df[kwargs['metric']]
            a_mean, a_std, a_means_list = fun_do_bootstrap(data_list)
            for _ in range(i):
                sample_data_list = list(np.random.choice(data_list, b_gr_size))
                b_mean, b_std, b_means_list = fun_do_bootstrap(sample_data_list)
                Z, pvalue = ztest(a_mean, a_std, b_mean, b_std)
                pvalue_list = pvalue_list + [pvalue]

        fun_draw_FPR_aa_test(ax_0, ax_1, pvalue_list)
        

def fun_get_df_in_ttest(a_var, b_var, a_len, b_len):
    """Рассчитывает степени свободы для двухвыборочного t-теста"""
    return (a_var/a_len + b_var/b_len)**2 / ( a_var**2/(a_len**2 * (a_len - 1)) + b_var**2/(b_len**2 * (b_len - 1)))  

def fun_draw_stat_grah(ax, stat, stat_mean, stat_std, df, stat_value, pvalue_2sided):
    """Рисуем на принимаемых графиках статистику
    
    Умеем рисовать нормальное и t распределения"""
    if stat == 'norm':
        distribution = stats.norm(loc = stat_mean, scale = stat_std)
    if stat == 't':
        distribution = stats.t(df = df, loc = stat_mean, scale = stat_std)
    x = np.linspace(stat_mean - stat_std*4, stat_mean + stat_std*4, 100)
    y = distribution.pdf(x)
    area_under_curve = distribution.sf(stat_value)
    ax.plot(x, y, label="PDF")
    ax.fill_between(x, 0, y, where=x>stat_value, alpha=0.3)
    ax.annotate(f"""
                p-value a>b = {area_under_curve:0.2f}
                p-value a!=b = {pvalue_2sided:0.2f}
                stat p = {stat_mean:0.3f}
                stat std = {stat_std:0.3f}
                    """, 
                   (stat_mean - stat_std*3,max(y)/2))

    
    
def fun_perfom_bootstrap(a_df, b_df, metric):
    """Проводим аа-тест и аб-тест"""
    a_list, b_list = a_df[metric], b_df[metric]
    
    fig, ax = plt.subplots(1,4)
    fun_perform_aa_test(ax[0], ax[1], 'bootstrap', a_df, len(b_list), metric = metric)
    
    a_mean, a_std, a_means_list = fun_do_bootstrap(a_list) 
    b_mean, b_std, b_means_list = fun_do_bootstrap(b_list) 
    stat_mean = b_mean - a_mean
    stat_std = np.sqrt(a_std**2 + b_std**2)
    
    x = np.linspace(np.min([a_mean, b_mean]) - max([a_std, b_std])*4, np.max([a_mean, b_mean]) + max([a_std, b_std])*4, 100)
    distribution_a = stats.norm(loc = a_mean , scale = a_std)
    y_a = distribution_a.pdf(x)
    distribution_b = stats.norm(loc = b_mean , scale = b_std)
    y_b = distribution_b.pdf(x)
    ax[2].plot(x,y_a,label = 'а: p=' + f'{a_mean:0.3f}' + ' std=' +  f"{a_std:0.3f}")
    ax[2].plot(x,y_b,label = 'b: p=' + f'{b_mean:0.3f}' + ' std=' +  f"{b_std:0.3f}")
    ax[2].hist(a_means_list, label = 'a bootstrap', alpha = 0.3, density = True)
    ax[2].hist(b_means_list, label = 'b bootstrap', alpha = 0.3, density = True)
    ax[2].legend()
    ax[2].set_xlabel('mean in group')
    ax[2].set_title('mean in ab-test')
    
    Z, pvalue = ztest(a_mean, a_std, b_mean, b_std)
    fun_draw_stat_grah(ax[3],'norm', stat_mean, stat_std, None, 0, pvalue)
    ax[3].set_xlabel('difference in p')
    ax[3].set_title('difference in p in ab-test')
    
    fig.suptitle('Bootstrap ' + metric)
    fig.set_size_inches(18,5)
    fig.show()

    
def fun_perfom_ttest(a_df, b_df, metric):
    """Проводим аа-тест и аб-тест"""
    a_list, b_list = a_df[metric], b_df[metric]
    
    a_mean, b_mean = np.mean(a_list), np.mean(b_list)
    a_var, b_var = np.var(a_list), np.var(b_list) #квадрат!
    a_std, b_std = np.sqrt(a_var), np.sqrt(b_var)
    a_len, b_len = len(a_list), len(b_list)
    stat_mean = b_mean - a_mean
    stat_var = (a_var/a_len + b_var/b_len)
    df = fun_get_df_in_ttest(a_var, b_var, a_len, b_len)
    
    fig, ax = plt.subplots(1,4)
    fun_perform_aa_test(ax[0], ax[1], 'ttest', a_df, len(b_list), metric = metric)
    
    x = np.linspace(np.mean([a_mean, b_mean]) - max([a_std, b_std])*4, np.mean([a_mean, b_mean]) + max([a_std, b_std])*4, 100)
    distribution_a = stats.t(df = a_len - 1, loc = a_mean , scale = a_std)
    y_a = distribution_a.pdf(x)
    distribution_b = stats.t(df = b_len - 1, loc = b_mean , scale = b_std)
    y_b = distribution_b.pdf(x)
    ax[2].plot(x,y_a,label = 'а: mean=' + f'{a_mean:0.3f}' + ' std=' +  f"{a_std:0.2f}")
    ax[2].plot(x,y_b,label = 'b: mean=' + f'{b_mean:0.3f}' + ' std=' +  f"{b_std:0.2f}")
    ax[2].legend()
    ax[2].set_xlabel('mean in group')
    ax[2].set_title('means in ab-test')
    
    T, p = stats.ttest_ind(a_list,b_list, equal_var=False)
    df = fun_get_df_in_ttest(a_var, b_var, a_len, b_len)
    fun_draw_stat_grah(ax[3],'t', stat_mean, np.sqrt(stat_var), df, 0, p)
    ax[3].set_xlabel('difference in mean')
    ax[3].set_title('difference in mean in ab-test')
    
    fig.suptitle('T-test '  + metric)
    fig.set_size_inches(18,5)


def fun_perfom_mannwhitneyu(a_df, b_df, metric):
    """Проводим аа-тест и аб-тест"""
    a_list, b_list = a_df[metric], b_df[metric]
    
    fig, ax = plt.subplots(1,4)
    fun_perform_aa_test(ax[0], ax[1], 'mannwhitneyu', a_df, len(b_list), metric = metric)
    
    # При справедливости нулевой гипотезы критерий имеет математическое ожидание 
    # {\displaystyle M(U)=n_{1}n_{2}/2}{\displaystyle M(U)=n_{1}n_{2}/2} 
    # и дисперсию {\displaystyle D(U)=n_{1}n_{2}(n_{1}+n_{2}+1)/12}{\displaystyle D(U)=n_{1}n_{2}(n_{1}+n_{2}+1)/12} 
    # и при достаточно большом объёме выборочных данных {\displaystyle (n_{1}>19,n_{2}>19)}{\displaystyle (n_{1}>19,n_{2}>19)} 
    # распределён практически нормально
    
    stat_mean = len(a_list) * len(b_list) / 2
    stat_std = np.sqrt(len(a_list)*len(b_list)*(len(a_list)+len(b_list)+1)/12)
    U, p = stats.mannwhitneyu(a_list, b_list, method="exact")
    fun_draw_stat_grah(ax[3],'norm', stat_mean, stat_std, None, U, p)
    ax[3].set_title('~ distribution U-statistic')
    ax[3].set_xlabel('U-statistic')
    
    fig.suptitle('Mann-Whitneyu test ' + metric)
    
    fig.set_size_inches(18,5)
    fig.show()
    
def ztest_for_share(a_num, a_den, b_num, b_den):
    n_1, n_2 = a_den, b_den
    p_1, p_2 = a_num/a_den, b_num/b_den
    std_1, std_2 = np.sqrt((p_1 * (1 - p_1)) / n_1) , np.sqrt((p_2 * (1 - p_2)) / n_2)
    X = stats.norm(loc = p_2 - p_1, scale = np.sqrt( std_1**2 + std_2**2 ))
    pvalue = 2*min([X.cdf(0), 1 - X.cdf(0)])
    Z = (p_2 - p_1)/np.sqrt( std_1**2 + std_2**2 )
    return Z, pvalue

def fun_perfom_ztest_for_global_share(a_df, b_df, numerator, denominator, metric_name):
    
    a_list = a_df.loc[:, [numerator, denominator]].values.tolist()
    b_list = b_df.loc[:, [numerator, denominator]].values.tolist()
    
    fig, ax = plt.subplots(1,4)
    
    fun_perform_aa_test(ax[0], ax[1], 'ztest_for_global_share', a_df, len(b_list), num = numerator, den = denominator)
    
    a_num, b_num = sum([i[0] for i in a_list ]), sum([i[0] for i in b_list ])
    a_den, b_den = sum([i[1] for i in a_list ]), sum([i[1] for i in b_list ])
    
    a_p, b_p = a_num/a_den, b_num/b_den 
    a_std, b_std = np.sqrt(a_p*(1-a_p)/a_den), np.sqrt(b_p*(1-b_p)/b_den)
    stat_p = b_p - a_p
    stat_std = np.sqrt(a_std**2 + b_std**2)
    
    x = np.linspace(np.min([a_p, b_p]) - max([a_std, b_std])*4, np.max([a_p, b_p]) + max([a_std, b_std])*4, 100)
    distribution_a = stats.norm(loc = a_p , scale = a_std)
    y_a = distribution_a.pdf(x)
    distribution_b = stats.norm(loc = b_p , scale = b_std)
    y_b = distribution_b.pdf(x)
    ax[2].plot(x,y_a,label = 'а: p=' + f'{a_p:0.3f}' + ' std=' +  f"{a_std:0.3f}")
    ax[2].plot(x,y_b,label = 'b: p=' + f'{b_p:0.3f}' + ' std=' +  f"{b_std:0.3f}")
    ax[2].legend()
    ax[2].set_xlabel('p in group')
    ax[2].set_title('p in ab-test')
    
    Z, pvalue = ztest_for_share(a_num, a_den, b_num, b_den)
    fun_draw_stat_grah(ax[3], 'norm', stat_p, stat_std, None, 0, pvalue)
    ax[3].set_xlabel('difference in p')
    ax[3].set_title('difference in p in ab-test')
    
    fig.suptitle('Z-test ' + metric_name)
    
    fig.set_size_inches(18,5)
    fig.show()
