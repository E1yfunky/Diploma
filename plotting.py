import openpyxl
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import plotly.express as px


def plot_results_by_nu(dimension, df, otn, way):
	mas = []
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '16'

	fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 24))

	suit_mas = []
	target_mas = []
	score_mas = []
	for nu in np.linspace(0, 3, 13):
		suit_mas.append(df[df.nu == nu].groupby(['iteration']).agg({'suitability': 'mean'})['suitability'])
		target_mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		score_mas.append(df[df.nu == nu].groupby(['iteration']).agg({'score': 'mean'})['score'])

	# ax[0].set_yscale('log')
	ax[0].boxplot(suit_mas)
	ax[0].set_xticklabels(np.linspace(0, 3, 13))
	ax[0].set_title(f'{dimension}d suitability of nu, 1:{otn}')
	ax[0].set_ylabel('Su', style='italic')
	ax[0].set_xlabel('$\\nu$', style='italic')

	# ax[1].set_yscale('log')
	ax[1].boxplot(target_mas)
	ax[1].set_xticklabels(np.linspace(0, 3, 13))
	ax[1].set_title(f'{dimension}d target of nu, 1:{otn}')
	ax[1].set_ylabel('$\widetilde{f}^*$')
	ax[1].set_xlabel('$\\nu$', style='italic')

	# ax[2].set_yscale('log')
	ax[2].boxplot(score_mas)
	ax[2].set_xticklabels(np.linspace(0, 3, 13))
	ax[2].set_title(f'{dimension}d score of nu, 1:{otn}')
	ax[2].set_ylabel('Sc', style='italic')
	ax[2].set_xlabel('$\\nu$', style='italic')

	plt.savefig(f"./results/{way}results_of_nu_lin.png")
	plt.show()


def plot_by_method(df, df_s, df_sc, dimension, way):
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '16'

	fig = plt.figure()
	ax = fig.add_subplot()

	target_mas = []
	target_mas.append(df.groupby(['seed']).agg({'target': 'min'})['target'])
	target_mas.append(df_s.groupby(['seed']).agg({'target': 'min'})['target'])
	target_mas.append(df_sc.groupby(['seed']).agg({'target': 'min'})['target'])

	ax.boxplot(target_mas)
	ax.set_xticklabels(['No', 'Suitability', 'Score'])
	ax.set_title(f'{dimension}d f of method')
	ax.set_ylabel('$\widetilde{f}^*$', style='italic')
	ax.set_xlabel('Method', style='italic')

	plt.savefig(f"./results/{way}results_of_method_lin.png")
	plt.show()


def plot_nu_by_method(df, df_s, df_sc, dimension, way):
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '16'

	fig = plt.figure()
	ax = fig.add_subplot()

	nu_mas = []
	nu_mas.append(df['nu'])
	nu_mas.append(df_s['nu'])
	for i in nu_mas[1]:
		if i == np.inf:
			i = 5

	nu_mas.append(df_sc['nu'])
	for i in nu_mas[2]:
		if i == np.inf:
			i = 5

	ax.boxplot(nu_mas)
	ax.set_xticklabels(['No', 'Suitability', 'Score'])
	ax.set_title(f'{dimension}d nu of method')
	ax.set_ylabel('nu', style='italic')
	ax.set_xlabel('Method', style='italic')

	plt.savefig(f"./results/{way}nu_of_method_lin.png")
	plt.show()


def the_best_of_mean_3d(df, dimension, way):
	mas = []
	f_mas = []
	nu_mas = []
	for nu in np.linspace(0, 3, 13):
		mas.extend(df[df.nu == nu].groupby(['seed']).agg({'suitability': 'mean'})['suitability'])
		f_mas.extend(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		nu_mas.extend([nu] * 10)
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(nu_mas, f_mas, mas, alpha=0.5)
	ax.set_title(f'{dimension}d f of m_suitability of nu')
	ax.set_zlabel('Su', style='italic')
	ax.set_ylabel('$\widetilde{f}^*$')
	ax.set_xlabel('$\\nu$', style='italic')

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

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig, axs = plt.subplots(nrows=4, ncols=4)
	fig.suptitle(f'{dimension}d f of m_suitability of nu')

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		axs[i % 4, i // 4].scatter(mas[i], f_mas[i], alpha=0.5)

	plt.savefig(f"./results/{way}f_of_m_suitability_for_nu_s.png")
	plt.show()


def score_history(df, dimension, way):
	s_mas = []
	max_t = df['iteration'].max()

	suit_mas = df.groupby(['iteration']).agg({'score': 'mean'})['score']
	max_s = -10
	temp_mas = []
	for suit in suit_mas:
		max_s = max(suit, max_s)
		temp_mas.append(max_s)
	s_mas = temp_mas

	#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.plot(np.arange(0, max_t + 1, 1), s_mas)
	ax.set_xlabel('t', style='italic')
	ax.set_ylabel('Sc', style='italic')
	# ax.legend()

	ax.set_title(f'{dimension}d history of score')
	plt.savefig(f"./results/{way}score_history.png")
	plt.show()


def suitability_history(df, dimension, way):
	s_mas = []
	max_t = df['iteration'].max()

	suit_mas = df.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	max_s = -10
	temp_mas = []
	for suit in suit_mas:
		max_s = max(suit, max_s)
		temp_mas.append(max_s)
	s_mas = temp_mas

	#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.plot(np.arange(0, max_t + 1, 1), s_mas)
	ax.set_xlabel('t', style='italic')
	ax.set_ylabel('Su', style='italic')
	# ax.legend()

	ax.set_title(f'{dimension}d history of suitability')
	plt.savefig(f"./results/{way}suitability_history.png")
	plt.show()


def suitability_history_comparison(df, df_s, df_sc, dimension, way):
	max_t = df['iteration'].max()

	suit_mas = df.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	max_s = -10
	temp_mas = []
	for suit in suit_mas:
		max_s = max(suit, max_s)
		temp_mas.append(max_s)
	s_mas = temp_mas

	suit_mas_s = df_s.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	max_s = -10
	temp_mas = []
	for suit in suit_mas_s:
		max_s = max(suit, max_s)
		temp_mas.append(max_s)
	s_mas_s = temp_mas

	suit_mas_sc = df_sc.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	max_s = -10
	temp_mas = []
	for suit in suit_mas_sc:
		max_s = max(suit, max_s)
		temp_mas.append(max_s)
	s_mas_sc = temp_mas

	#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.plot(np.arange(0, max_t + 1, 1), s_mas, 'red', label='No')
	ax.plot(np.arange(0, max_t + 1, 1), s_mas_s, 'blue', label='Suitability')
	ax.plot(np.arange(0, max_t + 1, 1), s_mas_sc, 'green', label='Score')
	ax.set_xlabel('t', style='italic')
	ax.set_ylabel('Su', style='italic')
	ax.legend()

	ax.set_title(f'{dimension}d history of suitability')
	plt.savefig(f"./results/{way}suitability_history.png")
	plt.show()


def f_history(df, dimension, way):
	f_mas = []
	max_t = df['iteration'].max()


	#for nu in np.linspace(0, 3, 13):
	target_mas = df.groupby(['iteration']).agg({'target': 'mean'})['target']
	min_f = 100000000000000
	temp_mas = []
	for target in target_mas:
		min_f = min(target, min_f)
		temp_mas.append(min_f)
	f_mas = temp_mas

	# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	#for i, nu in enumerate(np.linspace(0, 3, 13)):
	ax.plot(np.arange(0, max_t + 1, 1), f_mas)
	ax.set_xlabel('t', style='italic')
	ax.set_ylabel('$\widetilde{f}^*$')
	#ax.legend()

	ax.set_title(f'{dimension}d history of F')
	plt.savefig(f"./results/{way}f_history.png")
	plt.show()


def f_history_comparison(df, df_s, df_sc, dimension, way):
	max_t = df['iteration'].max()

	target_mas = df.groupby(['iteration']).agg({'target': 'mean'})['target']
	target_mas_s = df_s.groupby(['iteration']).agg({'target': 'mean'})['target']
	target_mas_sc = df_sc.groupby(['iteration']).agg({'target': 'mean'})['target']
	min_f = 100000000000000
	temp_mas = []
	for target in target_mas:
		min_f = min(target, min_f)
		temp_mas.append(min_f)
	f_mas = temp_mas

	temp_mas = []
	min_f = 100000000000000
	for target in target_mas_s:
		min_f = min(target, min_f)
		temp_mas.append(min_f)
	f_mas_s = temp_mas

	temp_mas = []
	min_f = 100000000000000
	for target in target_mas_sc:
		min_f = min(target, min_f)
		temp_mas.append(min_f)
	f_mas_sc = temp_mas

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.plot(np.arange(0, max_t + 1, 1), f_mas, 'red', label='No')
	ax.plot(np.arange(0, max_t + 1, 1), f_mas_s, 'blue', label='Suitabality')
	ax.plot(np.arange(0, max_t + 1, 1), f_mas_sc, 'green', label='Score')
	ax.set_xlabel('t', style='italic')
	ax.set_ylabel('$\widetilde{f}^*$')
	ax.legend()

	ax.set_title(f'{dimension}d history of F')
	plt.savefig(f"./results/{way}f_history.png")
	plt.show()


def score_suitability(df, df_s, df_sc, dimension, way):
	suit_mas = df.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	score_mas = df.groupby(['iteration']).agg({'score': 'mean'})['score']

	suit_mas_s = df_s.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	score_mas_s = df_s.groupby(['iteration']).agg({'score': 'mean'})['score']

	suit_mas_sc = df_sc.groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
	score_mas_sc = df_sc.groupby(['iteration']).agg({'score': 'mean'})['score']

	# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['font.size'] = '14'

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.plot(suit_mas, score_mas, 'r^', label='No')
	ax.plot(suit_mas_s, score_mas_s, 'b^', label='Suitability')
	ax.plot(suit_mas_sc, score_mas_sc, 'g^', label='Score')
	ax.set_xlabel('Su', style='italic')
	ax.set_ylabel('Score', style='italic')
	ax.legend()

	ax.set_title(f'{dimension}d history of suitability')
	plt.savefig(f"./results/{way}score_suitability_.png")
	plt.show()


def main():
	print("get started")
	dimension = 8
	otn = 3
	way = f'Ackley/{dimension}d/hypercube/'
	df = pd.read_csv(f"./results/{way}test_data_.csv",  delimiter=';')
	df_s = pd.read_csv(f"./results/{way}test_data_s.csv",  delimiter=';')
	df_sc = pd.read_csv(f"./results/{way}test_data_sc.csv",  delimiter=';')

	plot_nu_by_method(df, df_s, df_sc, dimension, way)
	plot_by_method(df, df_s, df_sc, dimension, way)
	score_suitability(df, df_s, df_sc, dimension, way)
	f_history_comparison(df, df_s, df_sc, dimension, way)
	suitability_history_comparison(df, df_s, df_sc, dimension, way)

	# plot_results_by_nu(dimension, df, otn, way)
	# f_history(df, dimension, way)
	# the_best_of_mean_2d(df, dimension, way)
	# the_best_of_mean_3d(df, dimension, way)
	# suitability_history(df, dimension, way)
	# score_history(df, dimension, way)


if __name__ == '__main__':
	main()