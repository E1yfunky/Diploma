import openpyxl
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import plotly.express as px

def main():
	print("get started")
	dimension = 2
	otn = 3
	df = pd.read_csv(f"./results/Rosenbrock/{dimension}d/1to{otn}/test_data.csv",  delimiter=';')

	mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['iteration']).agg({'suitability': 'mean'})['suitability'])

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.boxplot(mas)
	ax.set_xticklabels(np.linspace(0, 3, 13))
	ax.set_title(f'{dimension}d suitability of nu, 1:{otn}')
	ax.set_ylabel('suitability')
	ax.set_xlabel('Nu')

	plt.savefig(f"./results/Rosenbrock/{dimension}d/1to{otn}/suitability_of_nu.png")
	plt.show()


if __name__ == '__main__':
	main()