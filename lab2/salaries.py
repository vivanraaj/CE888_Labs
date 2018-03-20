import matplotlib
matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 


# def permutation(statistic, error):


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')
	print((df.columns))
        # drop columns which has nan values
        #df = df.dropna()
	sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

	sns_plot.axes[0,0].set_ylim(0,)
	#sns_plot.axes[0,0].set_xlim(0,)

	sns_plot.savefig("scaterplot_vehicle.png",bbox_inches='tight')
	sns_plot.savefig("scaterplot_vehicle.pdf",bbox_inches='tight')
	

	#################################################
	# create histogram for current fleet
	data_1 = df.values.T[0]
	
	#print((("Mean: %f")%(np.mean(data))))
	#print((("Median: %f")%(np.median(data))))
	#print((("Var: %f")%(np.var(data))))
	#print((("std: %f")%(np.std(data))))
	#print((("MAD: %f")%(mad(data))))

	plt.clf()
	sns_plot2 = sns.distplot(data_1, bins=20, kde=False, rug=True).get_figure()

	axes = plt.gca()
	axes.set_xlabel('Current Fleet') 
	axes.set_ylabel('MPG')

	sns_plot2.savefig("histogram_currentfleet.png",bbox_inches='tight')
	sns_plot2.savefig("histogram_currentfleet.pdf",bbox_inches='tight')
	
	
	########################################################
	# create histogram for current fleet
        # drop columns which has nan values
	df = pd.read_csv('./vehicles.csv')        
	df = df.dropna()	
	data_2 = df.values.T[1]
	
	#print((("Mean: %f")%(np.mean(data))))
	#print((("Median: %f")%(np.median(data))))
	#print((("Var: %f")%(np.var(data))))
	#print((("std: %f")%(np.std(data))))
	#print((("MAD: %f")%(mad(data))))

	plt.clf()
	sns_plot2 = sns.distplot(data_2, bins=20, kde=False, rug=True).get_figure()

	axes = plt.gca()
	axes.set_xlabel('New Fleet') 
	axes.set_ylabel('MPG')

	sns_plot2.savefig("histogram_newfleet.png",bbox_inches='tight')
	sns_plot2.savefig("histogram_newfleet.pdf",bbox_inches='tight')


	
