import os
import glob


def _list_models(path_neat,
        simulators=['nest', 'neuron'],
        pprint=True
    ):
    """
    Get (and print) a dictionary containing all installed model

    Parameters
    ----------
    path_neat:
        the path to the root directory of the import neat package
    simulators: list of str
        the simulators for which to show the model of a given name
    """
    models = {sim: [] for sim in simulators}

    if 'nest' in simulators:
        path_nest = os.path.join(path_neat, 'tools/simtools/', 'nest/tmp/*/')

        for file_path in glob.glob(path_nest):
            file_name = os.path.basename(os.path.normpath(file_path))
            # only append name if directory contains .nestml files
            path_test = os.path.join(file_path, "*.nestml")
            if not len([f for f in glob.glob(path_test)]) == 0:
                models['nest'].append(file_name)

    if 'neuron' in simulators:
        path_neuron = os.path.join(path_neat, 'tools/simtools/', 'neuron/tmp/*/')

        print(path_neuron)

        for file_path in glob.glob(path_neuron):
            file_name = os.path.basename(os.path.normpath(file_path))
            models['neuron'].append(file_name)

    if pprint:
        print("\n------- installed models --------")
        for simulator, model_list in models.items():
            print(f"> {simulator}")
            for model in model_list:
                print(f"  - {model}")
        print("---------------------------------\n")

    return models