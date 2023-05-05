try:
    import nest
    import nest.lib.hl_api_exceptions as nestexceptions
except ImportError as e:
    pass

import os
import subprocess

from neat import loadNeuronModel, loadNestModel


def load_or_install_neuron_testchannels():
    channel_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'channelcollection_for_tests.py'
    ))
    try:
        # loadNeuronModel() calls will raise a RuntimeError is a compiled model
        # is loaded multiple times
        try:
            # raises FileNotFoundError if not compiled
            loadNeuronModel("multichannel_test")
        except FileNotFoundError:
            subprocess.call([
                "neatmodels", "install", "multichannel_test",
                "-s", "neuron",
                "-p", channel_file
            ])
            loadNeuronModel("multichannel_test")
    except RuntimeError as e:
        # the neuron model "multichannel_test" has already been loaded
        pass


def load_or_install_nest_testchannels():
    channel_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'channelcollection_for_tests.py'
    ))
    try:
        # raises FileNotFoundError if not compiled
        loadNestModel("multichannel_test")
    except (nestexceptions.NESTErrors.DynamicModuleManagementError, FileNotFoundError):
        subprocess.call([
            "neatmodels", "install", "multichannel_test",
            "-s", "nest",
            "-p", channel_file
        ])
        loadNestModel("multichannel_test")

