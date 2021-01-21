"""
Script to generate a single training example for rotations of a hexagonal lattice.
"""
from bornagain import deg, angstrom, nm, nm2
import bornagain as ba

PHIMIN, PHIMAX = -1.07*deg, 1.07*deg
ALPHAMIN, ALPHAMAX = 0.0*deg, 2.2*deg
NPHI, NALPHA = 256, 256
BEAM_INTENSITY = 2e8


def get_default_par_values():
    default_values = {
        "lattice_constant": 34.0*nm,
        "peak_width": 150.0*nm,
        "pos_variance": 6.2*nm2,
        "sphere_radius": 10.0*nm,
        "wavelength": 1.341*angstrom,
        "beam_inclination": 0.164*deg,
    }
    return default_values


def get_par_values(par_values):
    result = get_default_par_values()
    if par_values:
        for name, val in par_values.items():
            result[name] = val
    return result


def get_sample(angle_distribution, par_values):
    """
    Returns a sample for given orientation distribution and parameter values.
    """
    # retrieve variable parameter values
    lattice_constant = par_values["lattice_constant"]
    peak_width = par_values["peak_width"]
    pos_variance = par_values["pos_variance"]
    radius = par_values["sphere_radius"]

    # defining materials
    m_ambience = ba.HomogeneousMaterial("Air", 0.0, 0.0)
    m_particle = ba.HomogeneousMaterial("CoFe2O4", 2.03e-5, 1.5e-6)
    m_layer = ba.HomogeneousMaterial("SiO2", 5.44e-6, 5.44e-8)
    m_substrate = ba.HomogeneousMaterial("Si", 5.78e-6, 1.02e-7)

    # layers
    air_layer = ba.Layer(m_ambience)
    oxide_layer = ba.Layer(m_layer, 60.0*nm)
    substrate_layer = ba.Layer(m_substrate)

    # particle and basic layout
    formfactor = ba.FormFactorFullSphere(radius)
    particle = ba.Particle(m_particle, formfactor)
    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(particle)

    decay_function = ba.FTDecayFunction2DCauchy(peak_width, peak_width, 0.0)

    # interference function and different layouts with correct weights
    for i, weight in enumerate(angle_distribution):
        angle = i*60.0 / len(angle_distribution)
        if weight > 0.0:
            interference = ba.InterferenceFunction2DLattice.createHexagonal(lattice_constant, angle*deg)
            interference.setDecayFunction(decay_function)
            interference.setPositionVariance(pos_variance)
            particle_layout.setInterferenceFunction(interference)
            particle_layout.setWeight(weight)
            air_layer.addLayout(particle_layout)

    multi_layer = ba.MultiLayer()
    multi_layer.addLayer(air_layer)
    multi_layer.addLayer(oxide_layer)
    multi_layer.addLayer(substrate_layer)

    return multi_layer


def get_simulation(par_values):
    """
    Returns GISAS simulation with given parameter values.
    """
    wavelength = par_values["wavelength"]
    inclination_angle = par_values["beam_inclination"]

    simulation = ba.GISASSimulation()
    simulation.setDetectorParameters(NPHI, PHIMIN, PHIMAX, NALPHA, ALPHAMIN, ALPHAMAX)
    
    simulation.setBeamParameters(wavelength, inclination_angle, 0.0*deg)
    simulation.setBeamIntensity(BEAM_INTENSITY)
    return simulation


def simulate(angle_distribution, par_values=None):
    """
    Runs simulation for given orientation distribution and parameter values.
    """
    new_par_values = get_par_values(par_values)

    sample = get_sample(angle_distribution, new_par_values)
    simulation = get_simulation(new_par_values)
    simulation.setSample(sample)
    simulation.runSimulation()
    return simulation.result()