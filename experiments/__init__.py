from experiments.experiment import Experiment
from experiments.sequence_model import SequencePokeModel
from experiments.fixed_length_model import FixedLengthModel


__experiments__ = {
    "sequence_poke_model": SequencePokeModel,
    "fixed_length_model": FixedLengthModel,
}


def select_experiment(config,dirs, device):
    experiment = config["general"]["experiment"]
    project_name = config["general"]["project_name"]
    if experiment not in __experiments__:
        raise NotImplementedError(f"No such experiment! {experiment}")
    if config["general"]["restart"]:
        print(f"Restarting experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    else:
        print(f"Running new experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    return __experiments__[experiment](config, dirs, device)
