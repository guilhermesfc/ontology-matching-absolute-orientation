import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
import datetime

def read_results(folder='../results/'):
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('csv') and len(f) == 28]
    
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(folder, f)))
        
    df = pd.concat(dfs)
    
    df.sort_values(by='timestamp', inplace=True)
    
    df['timestamp']= pd.to_datetime(df['timestamp'])
    df.reset_index(inplace=True)
    
    return df


def plot_precision(df,
                   col='alignment_noise',
                   x_label='alignment noise fraction',
                   ls='-', ax=None,
                   y_lim=(-5.0, 105.0)):

    with sns.plotting_context("notebook"):    

        if ax:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1,1)
            #fig.set_size_inches(10, 5)
        
        
        ax.plot(df[col], df['train-precision'], marker='o', ls=ls, label='train')
        ax.plot(df[col], df['test-precision'], marker='o', ls=ls, label='test')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('precision [%]')
        
        ax.legend()
        
        ax.grid()

        y_lim_pres = ax.get_ylim()
        if y_lim_pres[0] > y_lim[0] and y_lim_pres[1] < y_lim[1]:
            ax.set_ylim(y_lim)

    return fig


def plot_precision_3D(df, var_x='dimension', var_y='window', var_z='numberOfWalks', var_c='train-precision'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(10,10)
    
    x = df[var_x]
    y = df[var_y]
    z = df[var_z]
    c = df[var_c]
    
    img = ax.scatter(x, y, z, c=c, cmap=plt.get_cmap('Reds'))
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_zlabel(var_z)
    
    fig.colorbar(img)
    
    plt.show()



def print_best_model_config(df, top_n=10, apply_cond=True, cond_perc=0.05):
    cols = ['dimension', 'window', 'depth', 'numberOfWalks',
            'test-precision', 'train-precision',
            'norm_before_absOrient', 'norm_after_absOrient',
            'alignment_noise', 'train_percentage']
    
    condition = (df['train-precision'] > (1.-cond_perc) * df['test-precision'])\
              & (df['train-precision'] < (1.+cond_perc) * df['test-precision'])
    
    if apply_cond:
        df_best = df[condition][cols]
    else:
        df_best = df[cols]
    df_ret = df_best.sort_values(['test-precision', 'train-precision'], ascending=[False, False])[:top_n]
    
    return df_ret

def sel_best_model(df, an=0.0, apply_cond=True, cond_perc=0.05):
    df_40 =  print_best_model_config(df[
        (df.alignment_noise == an) & (df.train_percentage == 0.4)
    ], top_n=5, apply_cond=apply_cond, cond_perc=cond_perc)
    
    df_80 = print_best_model_config(df[
        (df.alignment_noise == an) & (df.train_percentage == 0.8)
    ], top_n=5, apply_cond=apply_cond, cond_perc=cond_perc)
    print(df_40[df_40.columns[:-2]])
    print(df_80[df_80.columns[:-2]])
    return (df_40.loc[df_40.index[0]].to_frame().T, df_80.loc[df_80.index[0]].to_frame().T)
