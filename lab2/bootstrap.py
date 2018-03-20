import matplotlib

matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def boostrap(statistic_func, iterations, data):
    samples = np.random.choice(data, replace=True, size=[iterations, len(data)])
    # print samples.shape
    data_mean = data.mean()

    vals = []
    for sample in samples:
        std = statistic_func(sample)
        # print sta
        vals.append(std)
    b = np.array(vals)
    # print b
    lower, upper = np.percentile(b, [2.5, 97.5])
    print lower, upper , data_mean
    return std, lower, upper


if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')
    # print df.columns
    data = df.values.T[0]
    #print data
    boots = []
    std_data = np.std(data)
    print ("Std_Current_Fleet: %f")%(np.std(data))
    print ("Mean_Current_Fleet: %f")%(np.mean(data))



    for i in range(100, 100000, 1000):
        boot = boostrap(np.mean, i, data)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
    sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_oldfleet.png", bbox_inches='tight')
    sns_plot.savefig("bootstrap_oldfleet.pdf", bbox_inches='tight')
    

    #################
    boots1 = []
    data1 = df.dropna().values.T[1]
    #print data1
    boots = []
    print ("Std_New_Fleet: %f")%(np.std(data1))
    print ("Mean_New_Fleet: %f")%(np.mean(data1))

    for i in range(100, 100000, 1000):
        boot1 = boostrap(np.mean, i, data1)
        boots1.append([i, boot[0], "mean"])
        boots1.append([i, boot[1], "lower"])
        boots1.append([i, boot[2], "upper"])

    df_boot_1 = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
    sns_plot = sns.lmplot(df_boot_1.columns[0], df_boot_1.columns[1], data=df_boot_1, fit_reg=False, hue="Value")

    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_newfleet.png", bbox_inches='tight')
    sns_plot.savefig("bootstrap_newfleet.pdf", bbox_inches='tight')




