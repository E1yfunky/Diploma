import openpyxl
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import random
import matplotlib.pyplot as plt

from itertools import product
from celluloid import Camera
from statistics import mean
from sklearn import linear_model
from bayesian_optimization import BayesianOptimization
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


def make_points_animation(optimizer, iters, x_range):
	fig = plt.figure()
	ax = fig.add_subplot()
	camera = Camera(fig)

	test_data = optimizer.test_x
	test_x = np.array([i[0] for i in test_data[:-iters]], dtype=float)
	test_points = optimizer.test_y[:-iters]

	x_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	z_ros = []
	for j in range(len(x_ros)):
		z_ros.append(
			list(get_rosenbrok_from_data(len(x_ros), 2, [[x_ros[j]] for i in range(len(x_ros))])))

	for i in range(len(test_points)):
		temp_x = test_x[:i + 1]
		temp_z = test_points[:i + 1]
		ax.scatter(temp_x, temp_z, color='green')
		ax.plot(x_ros, - np.array(z_ros), cmap='inferno')
		camera.snap()

	animation = camera.animate()
	animation.save('my_2animation.gif')


def make_3d_points_animation(optimizer, iters, x_range):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	camera = Camera(fig)


	test_data = optimizer.test_x
	test_data = np.array([np.array([key for key in i], dtype=float) for i in test_data[:-iters]])
	test_x = np.array([i[0] for i in test_data], dtype=float)
	test_y = np.array([i[1] for i in test_data], dtype=float)
	test_points = optimizer.test_y[:-iters]

	x_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	y_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	x_ros, y_ros = np.meshgrid(x_ros, y_ros)
	z_ros = []
	for j in range(len(x_ros)):
		z_ros.append(list(get_rosenbrok_from_data(len(x_ros[j]), 2, [[x_ros[j][i], y_ros[j][i]] for i in range(len(x_ros[j]))])))

	for i in range(len(test_points)):
		temp_x = test_x[:i + 1]
		temp_y = test_y[:i + 1]
		temp_z = test_points[:i + 1]

		ax.scatter(temp_x, temp_y, temp_z, color='green')
		ax.plot_wireframe(x_ros, y_ros, - np.array(z_ros), cmap='inferno')
		camera.snap()

	animation = camera.animate()
	animation.save('my_animation.gif')


def get_der(y1, y2, h):
	return (y2 - y1) / h


def add_rosenbrok_data(n, m, X, y):
	for j in range(0, m):
		temp_X = []
		for i in range(0, n):
			temp_X.append(4 * random.random())
		X = np.append(X, [temp_X], axis=0)

	for j in range(m):
		temp_y = 0
		for i in range(n - 1):
			temp_y += (1 - X[j][i]) ** 2 + 100 * ((X[j][i + 1] - X[j][i] ** 2) ** 2)
		y = np.append(y, [temp_y])

	return X, y


def get_rosenbrok_data(n, m):
	"""
	Возвращает начальные случайные m соответствий параметров-значений
	размерность вектора параметров n
	:param n: размерность вектора параметров (целое)
	:param m: число точек (целое)
	:return: X - двумерный массив случайных точек размерности n,
	y - одномерный массив значений функции
	"""
	X = np.empty((m, n))
	y = np.empty((m,))
	for j in range(0, m):
		temp_X = []
		for i in range(0, n):
			temp_X.append(4 * random.random())
		X[j] = np.asarray(temp_X)

	for j in range(m):
		temp_y = 0
		for i in range(n - 1):
			temp_y += (1 - X[j][i]) ** 2 + 100 * ((X[j][i + 1] - X[j][i] ** 2) ** 2)
		y[j] = (np.asarray([temp_y]))

	return X, y


def get_rosenbrok_from_data(m, n, X):
	"""
	Возвращает m результатов функции Розенброка для заданных точек X размерностей n
	:param m: число точек (целое)
	:param n: размерность вектора параметров (целое)
	:param X: двумерный массив точек
	:return: одномерный массив значений функции
	"""
	y = np.empty((m,))
	for j in range(0, m):
		temp_y = 0
		for i in range(0, n - 1):
			temp_y += (1 - X[j][i]) ** 2 + 100 * ((X[j][i + 1] - X[j][i] ** 2) ** 2)
		y[j] = (np.array([- temp_y]))

	return y


def add_srinivas_data(n, m, X, y):
	for j in range(0, m):
		temp_X = []
		temp_y = 0
		for i in range(0, n):
			temp_X.append(2 * random.random() - 1)
			temp_y += np.sin(np.pi * temp_X[i]) / np.pi * temp_X[i]
		X = np.append(X, [temp_X], axis=0)
		y = np.append(y, [temp_y])

	return X, y


def get_srinivas_data(n, m):
	"""
	Возвращает начальные случайные m соответствий параметров-значений
	размерность вектора параметров n
	:param n: размерность вектора параметров (целое)
	:param m: число точек (целое)
	:return: X - двумерный массив случайных точек размерности n,
	y - одномерный массив значений функции
	"""
	X = np.empty((m, n))
	y = np.empty((m,))
	for j in range(0, m):
		temp_X = []
		temp_y = 0
		for i in range(0, n):
			temp_X.append(2 * random.random() - 1)
			temp_y += np.sin(np.pi * temp_X[i]) / np.pi * temp_X[i]
		X[j] = np.asarray(temp_X)
		y[j] = (np.asarray([temp_y]))

	return X, y


def get_srinivas_from_data(m, n, X):
	"""
	Возвращает m результатов функции Сриниваса для заданных точек X размерностей n
	:param m: число точек (целое)
	:param n: размерность вектора параметров (целое)
	:param X: двумерный массив точек
	:return: одномерный массив значений функции
	"""
	y = np.empty((m,))
	for j in range(0, m):
		temp_X = []
		temp_y = 0
		for i in range(0, n):
			temp_X.append(X[j][i])
			temp_y += np.sin(np.pi * temp_X[i]) / np.pi * temp_X[i]
		y[j] = (np.asarray([temp_y]))

	return y


def black_box_func(**X):
	X = np.array([X[key] for key in sorted(X)], dtype=float)

	y = 0
	for i in range(len(X) - 1):
		y += (1 - X[i]) ** 2 + 100 * ((X[i + 1] - X[i] ** 2) ** 2)
	return -y


def black_box_func_1(**X):
	X = np.array([X[key] for key in sorted(X)], dtype=float)
	y = 0
	for i in range(len(X)):
		y += np.sin(np.pi * X[i]) / np.pi * X[i]
	return -y


def bayes_optim(d, nu_mas, init_points, n_iter, x_range, n, true_res):
	result_data = []
	df_dct = {'f_name': ['Rosenbrock'] * n_iter * len(nu_mas) * n,
			  'dimension': [d] * n_iter * n * len(nu_mas),
			  'nu': [],
			  'iteration': [i for i in range(n_iter)] * len(nu_mas) * n,
			  'init_points': [init_points] * n_iter * len(nu_mas) * n,
			  'iter_s': [n_iter / init_points] * n_iter * len(nu_mas) * n,
			  'X': [],
			  'target': [],
			  'model': [],
			  'score': [],
			  'suitability': [],
			  'seed': []}
	for nu in nu_mas:
		for i in range(n):
			seed = random.randint(1, 30000)
			df_dct['nu'].extend([nu] * n_iter)
			optimizer = BayesianOptimization(f=black_box_func,
											 pbounds={f"x[{_}]": x_range for _ in range(d)},
											 test_f=get_rosenbrok_from_data,
											 verbose=2,
											 random_state=seed,
											 nu=nu)

			optimizer.maximize(init_points=init_points, n_iter=n_iter)
			df_dct['X'].extend(optimizer.test_x)
			df_dct['target'].extend(optimizer._res)
			df_dct['model'].extend(optimizer._model_res)
			df_dct['score'].extend(optimizer._score_res)
			df_dct['suitability'].extend(optimizer._suit)
			df_dct['seed'].extend([seed]*n_iter)
			if len(result_data) > 0 and len(result_data[-1]) % n > 0:
				result_data[-1].append(-optimizer.max["target"])
			else:
				result_data.append([-optimizer.max["target"]])
			print(optimizer._suit)
			print(result_data)
			print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
		# print(df_dct)

	return nu_mas, np.array(result_data), df_dct


def main():
	random.seed(7)

	func = "Rosenbrock"
	df_dct = {'f_name': [],
			  'dimension': [],
			  'init_points': [],
			  'iter_s': [],
			  'nu': [],
			  'iteration': [],
			  'X': [],
			  'target': [],
			  'model': [],
			  'score': [],
			  'suitability': [],
			  'seed': []}

	x_range = [-1, 2]
	min_nu = 0
	max_nu = 3
	nu_mas = np.linspace(min_nu, max_nu, 13)
	d_dct = {2: 12, 4: 80, 8: 180}

	for i, d, points in enumerate(d_dct.items()):
		n_inter = 2 * points
		X, y_s, temp_df_dct = bayes_optim(d, nu_mas, points, n_inter, x_range, 10, 0.0)
		for key in df_dct.keys():
			df_dct[key].extend(temp_df_dct[key])

		df_marks = pd.DataFrame(temp_df_dct)

		writer = pd.ExcelWriter(f'./results/{func}/{d}d/test_data.xlsx')
		df_marks.to_excel(writer)
		writer.save()
		print('DataFrame is written successfully to Excel File.')

		df_marks.to_csv(f'./results/{func}/{d}d/test_data.csv', header=True, sep=';')
		print('DataFrame is written successfully to csv.')

	df_func = pd.DataFrame(df_dct)

	writer = pd.ExcelWriter(f'./results/{func}/test_data.xlsx')
	df_func.to_excel(writer)
	writer.save()
	print('DataFrame is written successfully to Excel File.')

	df_func.to_csv(f'./results/{func}/test_data.csv', header=True, sep=';')
	print('DataFrame is written successfully to csv.')

		# df_marks.to_pickle(f'./results/{func}/{d}d/test_data.pkl')
		# print('DataFrame is written successfully to pkl.')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
