import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DefaultPhysiology:
    # default concentrations (same as NEURON)
    conc: Dict[str, float] = field(
        default_factory=lambda: {'na': 10., 'k': 54.4, 'ca': 1e-4} # mM
    )
    # default ion channel reversals
    e_rev: Dict[str, float] = field(
        default_factory=lambda: {'na': 50., 'k': -85., 'ca': 50.} # mV
    )
    # default temperature
    temp: float = 36. # celcius


@dataclass
class DefaultFitting:
    # holding potentials channel fit
    e_hs: np.array = field(
        default_factory=lambda: np.array([-75., -55., -35., -15.])
    )
    # holding concentrations channel fit
    conc_hs: Dict[str, np.array] = field(
        default_factory=lambda: {'ca': np.array([0.00010, 0.00012, 0.00014, 0.00016])}
    )

    # holding potentials concmech fit
    e_hs_cm: np.array = field(
        default_factory=lambda: np.array([-78.22, -68.22, -58.22,])
    )
    # holding concentrations concmech fit
    conc_hs_cm: Dict[str, np.array] = field(
        default_factory=lambda: {'ca': np.array([0.000100, 0.000105, 0.000110])}
    )

    # frequency evaluation
    freqs: float = 0. # Hz
    freqs_tau: np.array = field(
        default_factory=lambda: np.logspace(0., 3., 100)*1j # Hz
    )

    # time-points at which to evaluate time domain kernels for capacitance fit
    t_fit: np.array = field(
        default_factory=lambda: np.array([
            11., 12., 13., 14., 15.,
            16., 17., 18., 19., 20.,
            22., 24., 26., 28. ,30.,
        ]) # ms
    )


@dataclass
class DefaultMechParams:
    # concentration mechanism default parameters
    exp_conc_mech: Dict[str, float] = field(
        default_factory=lambda: {'gamma': 0., 'tau': 100.} # [tau] = ms
    )