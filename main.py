import numpy as np
import random


def Pdf_gaussian(m, sigma, x):
    return (1 / (np.sqrt(sigma) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / np.sqrt(sigma)) ** 2)


def delta(mean, sigma, num):
    return np.random.normal(loc=mean, scale=sigma, size=num)


def get_weights(oly, ars, pav, asc, samples):
    difference = abs(np.linalg.norm(samples - OLY_loc, axis=1) - oly) + abs(np.linalg.norm(samples - ARS_loc, axis=1) - ars) + abs(np.linalg.norm(
        samples - PAV_loc, axis=1) - pav) + abs(np.linalg.norm(samples - ASC_loc, axis=1) - asc)
    return Pdf_gaussian(8, 4, difference)


def resample(weights, samples):
    sort_weights_index = weights.argsort()
    sort_weights_index = sort_weights_index[2500:]
    new_samples = samples[sort_weights_index]
    new_weight = weights[sort_weights_index]
    new_half = random.choices(population=new_samples, weights=new_weight, k=2500)
    return np.concatenate((new_samples, new_half))


oly = []
ars = []
pav = []
asc = []
OLY_loc = np.array([-133, 18])
ARS_loc = np.array([-121, -9])
PAV_loc = np.array([-113, 1])
ASC_loc = np.array([-104, 12])
input()
for i in range(20):
    oly.append(float(input()))
input()
for i in range(20):
    ars.append(float(input()))
input()
for i in range(20):
    pav.append(float(input()))
input()
for i in range(20):
    asc.append(float(input()))

x_sample = np.random.uniform(-170, -90, 5000)
y_sample = np.random.uniform(-20, 40, 5000)
samples = np.stack((x_sample, y_sample), axis=-1)
weights = []
for i in range(20):
    x_move = delta(2, 1, 5000)
    y_move = delta(1, 1, 5000)
    moves = np.stack((x_move, y_move), axis=-1)
    samples = samples + moves  # elapse
    weights = get_weights(oly[i], ars[i], pav[i], asc[i], samples)  # generate weights
    samples = resample(weights, samples)

x, y = np.average(samples, axis=0, weights=weights)

print(int(np.ceil(x / 10) * 10))
print(int(np.ceil(y / 10) * 10))
