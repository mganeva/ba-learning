"""
Script to generate training data for rotations of a hexagonal lattice
"""
import random, json
import numpy as np
import bornagain as ba
import single_lattice as sl
import sys, os
from bornagain import deg, angstrom, nm, nm2
from os.path import join
from time import time


def unit_factor(unit_name):
    unit_table = {
        "deg": deg,
        "angstrom": angstrom,
        "nm": nm,
        "nm2": nm2
    }
    if unit_name in unit_table:
        return unit_table[unit_name]
    return 1.0


def convert_parameter_ranges(parameters):
    result = {}
    for name, val in parameters.items():
        if (len(val) > 2):
            unit = unit_factor(val[2])
        else:
            unit = 1.0
        result[name] = [ val[0]*unit, val[1]*unit ]
    return result


def get_variable_sim_par_values(config):
    par_limits = convert_parameter_ranges(config["parameters"])
    par_values = {}
    for name, limits in par_limits.items():
        width = limits[1] - limits[0]
        value = limits[0] + width * random.random()
        par_values[name] = value
    return par_values


def sim_distr(angle_distribution, save_path, index, config):
    """
    Runs a simulation for a given distribution of lattice angles and saves
    the intensity, distribution and external parameters (numpy + json)
    """
    par_values = get_variable_sim_par_values(config)
    result = sl.simulate(angle_distribution, par_values)
    filename_array = "data_distr{0}".format(index)
    filename_distr = "distr{0}".format(index)
    filename_params = "ext_pars{0}.json".format(index)
    np.save(join(save_path, filename_array), result.array())
    np.save(join(save_path, filename_distr), angle_distribution)
    with open(join(save_path, filename_params), 'w') as fp:
        json.dump(par_values, fp, indent=4)


def generate_angle_distribution(num_bins, max_non_zeros):
    """
    Generate a discrete probability distribution over num_bins values
    """
    distr = np.zeros(num_bins)
    non_zeros = random.randrange(max_non_zeros) + 1
    succes = False
    while not succes:
        nonzero_indices = random.sample(range(num_bins), non_zeros)
        probs = np.random.random_sample((non_zeros,))
        total = np.sum(probs)
        if total > 0.0:
            probs = probs / total
            succes = True
        for i, idx in enumerate(nonzero_indices):
            distr[idx] = probs[i]
    return non_zeros, distr

def generate_dataset(n_train, save_path, config):
    """
    Generates multiple simulations and stores the results on disk
    """
    N_ANGLES = int(config["n_angles"])
    MAX_NONZEROS = int(config["max_nonzeros"])
    for i in range(n_train):
        non_zeros, distr = generate_angle_distribution(N_ANGLES, MAX_NONZEROS)
        if i%10 == 9:
            status = "Example {0:d}: {1:d} non-zero probabilities".format(i+1, non_zeros)
            print(status)
        sim_distr(distr, save_path, i, config)


def create_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.isdir(path):
        raise FileExistsError(path)


def seed_random_generators():
    random.seed(300416)
    np.random.seed(240201)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: gen_lattice.py config_file target_directory")
        sys.exit(0)

    with open(sys.argv[1], 'r') as fp:
        config = json.load(fp)

    seed_random_generators()

    # Initialize paths to store data
    DATA_PATH = sys.argv[2]
    TRAIN_PATH = DATA_PATH + "/train"
    VAL_PATH = DATA_PATH + "/val"

    create_dir(DATA_PATH)
    create_dir(TRAIN_PATH)
    create_dir(VAL_PATH)

    if os.listdir(TRAIN_PATH):
        print("WARNING: '{}' is not empty! Files may be overwritten"
              .format(TRAIN_PATH))

    if os.listdir(VAL_PATH):
        print("WARNING: '{}' is not empty! Files may be overwritten"
              .format(VAL_PATH))

    with open(DATA_PATH + "/config.json", 'w') as fp:
        json.dump(config, fp, indent=4)

    # Data generation for training/validation
    N_TRAIN = int(config["n_train"])
    N_VAL = int(config["n_validation"])

    print("Generating training data: {0:d} examples".format(N_TRAIN))
    START_TIME = time()
    generate_dataset(N_TRAIN, TRAIN_PATH, config)
    print("Execution time for {0} training examples: {1:.2f} seconds"
          .format(N_TRAIN, time() - START_TIME))

    print("Generating validation data: {0:d} examples".format(N_VAL))
    START_TIME = time()
    generate_dataset(N_VAL, VAL_PATH, config)
    print("Execution time for {0} validation examples: {1:.2f} seconds"
          .format(N_VAL, time() - START_TIME))