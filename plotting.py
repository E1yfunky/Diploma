import openpyxl
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import plotly.express as px


def plot_f_by_nu(dimension, df, otn, way):
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

	plt.savefig(f"./results/{way}target_of_nu.png")
	plt.show()


def plot_suitability_by_nu(dimension, df, otn, way):
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

	plt.savefig(f"./results/{way}suitability_of_nu.png")
	plt.show()


def the_best_of_mean_3d(df, dimension, way):
	mas = []
	f_mas = []
	nu_mas = []
	for nu in np.linspace(0, 3, 13):
		mas.extend(df[df.nu == nu].groupby(['seed']).agg({'suitability': 'mean'})['suitability'])
		f_mas.extend(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		nu_mas.extend([nu] * 10)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(nu_mas, f_mas, mas, alpha=0.5)
	ax.set_title(f'{dimension}d f of m_suitability of nu')
	ax.set_zlabel('suitability')
	ax.set_ylabel('F*')
	ax.set_xlabel('Nu')

	plt.savefig(f"./results/{way}f_of_m_suitability_of_nu.png")
	plt.show()


def the_best_of_mean_2d(df, dimension, way):
	mas = []
	f_mas = []
	nu_mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['seed']).agg({'suitability': 'mean'})['suitability'])
		f_mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		nu_mas.append([nu] * 10)

	fig, axs = plt.subplots(nrows=4, ncols=4)
	fig.suptitle(f'{dimension}d f of m_suitability of nu')

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		axs[i // 4, i % 4].scatter(mas[i], f_mas[i], alpha=0.5)

	plt.savefig(f"./results/{way}f_of_m_suitability_for_nu_s.png")
	plt.show()


def suitability_history(df, dimension, way):
	s_mas = []
	max_t = df['iteration'].max()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		max_s = -1
		temp_mas = []
		seed = df[df.nu == nu]['seed'][10 * i * (max_t + 1)]
		for j in range(max_t + 1):
			max_s = max(df[(df.nu == nu) & (df.iteration == j) & (df.seed == seed)]['suitability'][10 * i * (max_t + 1) + j], max_s)
			temp_mas.append(max_s)
		s_mas.append(temp_mas)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	fig = plt.figure()
	ax = fig.add_subplot()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		ax.plot(np.arange(0, max_t + 1, 1), s_mas[i], color=colors[i], label=f'nu = {nu}')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Suitability')
	ax.legend()

	ax.set_title(f'{dimension}d history of suitability')
	plt.savefig(f"./results/{way}suitability_history.png")
	plt.show()


def f_history(df, dimension, way):
	f_mas = []
	max_t = df['iteration'].max()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		min_f = 100000000000000
		temp_mas = []
		seed = df[df.nu == nu]['seed'][10 * i * (max_t + 1)]
		for j in range(max_t + 1):
			min_f = min(df[(df.nu == nu) & (df.iteration == j) & (df.seed == seed)]['target'][10 * i * (max_t + 1) + j], min_f)
			temp_mas.append(min_f)
		f_mas.append(temp_mas)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	fig = plt.figure()
	ax = fig.add_subplot()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		ax.plot(np.arange(0, max_t + 1, 1), f_mas[i], color=colors[i], label=f'nu = {nu}')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('F*')
	ax.legend()

	ax.set_title(f'{dimension}d history of F')
	plt.savefig(f"./results/{way}f_history.png")
	plt.show()


def main():
	print("get started")
	dimension = 4
	otn = 3
	way = f'Rosenbrock/{dimension}d/hypercube/not_centered/'
	df = pd.read_csv(f"./results/{way}test_data.csv",  delimiter=';')

	the_best_of_mean_2d(df, dimension, way)
	the_best_of_mean_3d(df, dimension, way)
	suitability_history(df, dimension, way)
	plot_suitability_by_nu(dimension, df, otn, way)
	plot_f_by_nu(dimension, df, otn, way)
	f_history(df, dimension, way)


if __name__ == '__main__':
	main()