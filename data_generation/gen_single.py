import random, json, sys
import numpy as np
import bornagain as ba
import single_lattice as sl
from bornagain import deg, angstrom, nm, nm2
from os.path import join


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


def generate_single_sim(angle_distribution, save_path, config):
    par_values = get_variable_sim_par_values(config)
    result = sl.simulate(angle_distribution, par_values)
    filename_array = "data"
    filename_params = "ext_pars.json"
    np.save(join(save_path, filename_array), result.array())
    with open(join(save_path, filename_params), 'w') as fp:
        json.dump(par_values, fp, indent=4)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: gen_single.py config_file")
        sys.exit(0)

    with open(sys.argv[1], 'r') as fp:
        config = json.load(fp)
    
    # use a uniform distribution
    angle_distribution = np.full(120, 1./120)
    generate_single_sim(angle_distribution, ".", config)