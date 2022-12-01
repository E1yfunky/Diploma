import openpyxl
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import plotly.express as px


def plot_f_by_nu(dimension, df, otn):
	mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.boxplot(mas)
	ax.set_xticklabels(np.linspace(0, 3, 13))
	ax.set_title(f'{dimension}d target of nu, 1:{otn}')
	ax.set_ylabel('F*')
	ax.set_xlabel('Nu')

	plt.savefig(f"./results/Rosenbrock/{dimension}d/hypercube/not_centered/target_of_nu.png")
	plt.show()


def plot_suitability_by_nu(dimension, df, otn):
	mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['iteration']).agg({'suitability': 'mean'})['suitability'])

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.boxplot(mas)
	ax.set_xticklabels(np.linspace(0, 3, 13))
	ax.set_title(f'{dimension}d suitability of nu, 1:{otn}')
	ax.set_ylabel('Suitability')
	ax.set_xlabel('Nu')

	plt.savefig(f"./results/Rosenbrock/{dimension}d/hypercube/not_centered/target_of_nu.png")
	plt.show()


def main():
	print("get started")
	dimension = 2
	otn = 3
	df = pd.read_csv(f"./results/Rosenbrock/{dimension}d/hypercube/not_centered/test_data.csv",  delimiter=';')

	mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.boxplot(mas)
	ax.set_xticklabels(np.linspace(0, 3, 13))
	ax.set_title(f'{dimension}d target of nu, 1:{otn}')
	ax.set_ylabel('F*')
	ax.set_xlabel('Nu')

	plt.savefig(f"./results/Rosenbrock/{dimension}d/hypercube/not_centered/target_of_nu.png")
	plt.show()


if __name__ == '__main__':
	main()