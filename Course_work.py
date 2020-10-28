import math
import sys
from scipy.stats import chi2
import numpy as np
import scipy.stats as ss

from collections import namedtuple

Range = namedtuple('Range', ['left', 'right'])

DISTRIBUTION_SIZE = 1000
MIN_AMOUNT_IN_RANGE = 5


class DistributionFunction:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ss.norm.cdf(x, loc=self.mean, scale=self.std)


def generate_normal_distribution(size):
    return np.random.uniform(-sqrt(3), sqrt(3), (1, size))[0]


def generate_range_frequency_probability_lists(distr, empirical_distr_func):
    def fill_range_list(min, max, cur_step):
        num = int((max - min) / cur_step)
        result = list()
        result.append(Range(-float('inf'), min + cur_step))
        for i in range(0, num - 2):
            result.append(Range(result[i].right, result[i].right + cur_step))
        result.append(Range(result[num - 2].right, float("inf")))

        return result

    def fill_probability_list(range_list, distr_func):
        result = list()
        for range in range_list:
            result.append(distr_func(range.right) - distr_func(range.left))
        return result

    def fill_frequency_list(range_list, distr_list):
        result = list()
        for range in range_list:
            freq = 0
            for val in distr_list:
                if range.left <= val <= range.right:
                    freq += 1
            result.append(freq)
        return result

    def check(size, prob_list):
        for i in range(0, len(prob_list)):
            if size * prob_list[i] < MIN_AMOUNT_IN_RANGE:
                return False
        return True

    size = len(distr)
    min_x = min(distr)
    max_x = max(distr)

    step = (max_x - min_x) / size

    cur_step = step
    should_stop = False
    while not should_stop:
        range_list = fill_range_list(min_x, max_x, cur_step)
        probability_list = fill_probability_list(range_list, empirical_distr_func)
        frequency_list = fill_frequency_list(range_list, distr)

        if not check(size, probability_list):
            cur_step += step
        else:
            should_stop = True

    return range_list, probability_list, frequency_list


def get_distribution_parameters(distr):
    mu = np.mean(distr)
    sum = 0
    for i in range(0, len(distr)):
        sum += math.pow(distr[i] - mu, 2)
    squared_sigma = sum / len(distr)

    return mu, math.sqrt(squared_sigma)


def calculate_pirson_criterion(distr_size, range_list, prob_list, freq_list):
    criterion = 0
    all_freq = 0
    all_prob = 0
    all_np = 0
    all_n_i_minus_np = 0
    for i in range(0, len(range_list)):
        np_i = distr_size * prob_list[i]
        n_i_minus_np_i = freq_list[i] - np_i
        coef = math.pow(n_i_minus_np_i, 2) / np_i

        criterion += coef
        all_freq += freq_list[i]
        all_prob += prob_list[i]
        all_np += np_i
        all_n_i_minus_np += n_i_minus_np_i
    return criterion


"""
        print("%i; %.4f - %.4f\t %i\t %.4f\t %.4f\t %.4f\t %.4f" %
              (i + 1,
               range_list[i].left, range_list[i].right,
               freq_list[i],
               prob_list[i],
               np_i,
               n_i_minus_np_i,
               coef
               ))

    print(";;%i;%.4f;%.4f;%.4f" % (all_freq, all_prob, all_np, all_n_i_minus_np))
"""

k_list = []

true_norm = 0
false_norm = 0

for i in range(100):

    distr = generate_normal_distribution(DISTRIBUTION_SIZE)

    mu, sigma = get_distribution_parameters(distr)
    empirical_distr_func = DistributionFunction(mu, sigma)

    range_list, prob_list, freq_list = generate_range_frequency_probability_lists(distr, empirical_distr_func)
    k = len(range_list)

    # f = open('out100.csv', 'w')

    criterion = calculate_pirson_criterion(len(distr), range_list, prob_list, freq_list)

    # print("%.4f; %.4f" % (k, criterion))
    p = 0.95
    df = k - 1
    value = chi2.ppf(p, df)
    if criterion < value:
        true_norm = true_norm + 1
    else:
        false_norm = false_norm + 1

    print(k, criterion, value)
# print(set(k_list))
print(true_norm, false_norm)

import matplotlib.pyplot as plt

fig = plt.figure()

fig, ax = plt.subplots()

x1 = [10, 100, 1000]
y1 = [100, 98, 97]
ax.scatter(x1, y1)

ax.set(title='Количество успешных распознаваний нормального распределения\n по критерию хи-квадрат')
ax.plot(x1, y1)
ax.set_xlabel('Мощность выборки')
ax.set_ylabel('Мощность выборки')
plt.show()

import numpy as np
import random
from math import sqrt

np.random.uniform(-sqrt(3), sqrt(3), (1, 10))[0]

import matplotlib.pyplot as plt

fig = plt.figure()

fig, ax = plt.subplots()

x1 = [10, 100, 1000]
y1 = [100, 73, 0]

ax.scatter(x1, y1)

ax.set(title='Количество успешных распознаваний равномерного распределения\n по критерию хи-квадрат')
ax.plot(x1, y1)
ax.set_xlabel('Мощность выборки')
ax.set_ylabel('Мощность выборки')
plt.show()