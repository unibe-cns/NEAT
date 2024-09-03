import os
import shutil


def _check_model_name(model_name):
    if not len(model_name) > 0:
        raise IOError(
            "No model name [name] argument was provided. " "Nothing to uninstall."
        )

    if "/" in model_name or "." in model_name:
        raise IOError(
            "Model name [name] is a path name (contains '/') or "
            "a file name (contains '.', which is not allowed."
        )


def _uninstall_models(
    *model_names,
    path_neat,
    simulators=["nest", "neuron"],
):
    """
    Uninstall the model with the given name from the provided simulators

    Parameters
    ----------
    model_names: str
        the name of the model to be uninstalled
    path_neat:
        the path to the root directory of the import neat package
    simulators: list of str
        the simulators for which to uninstall the model of a given name
    """
    for model_name in model_names:
        _check_model_name(model_name)

        if "nest" in simulators:
            try:
                path_nest = os.path.join(
                    path_neat, "simulations/", f"nest/tmp/{model_name}/"
                )
                shutil.rmtree(path_nest)
                print(f"> Uninstalled {model_name} from nest")
            except FileNotFoundError as e:
                print(f"> {model_name} not found in nest, nothing to uninstall.")

        if "neuron" in simulators:
            try:
                path_neuron = os.path.join(
                    path_neat, "simulations/", f"neuron/tmp/{model_name}/"
                )
                shutil.rmtree(path_neuron)
                print(f"> Uninstalled {model_name} from neuron")
            except FileNotFoundError as e:
                print(f"> {model_name} not found in neuron, nothing to uninstall.")
