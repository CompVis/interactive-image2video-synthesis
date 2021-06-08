import argparse
from os import path
import yaml
import os

from experiments import select_experiment



def create_dir_structure(model_name, experiment):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join("/export/scratch3/ablattma/visual_poking/",experiment,subdir,model_name) for subdir in subdirs}
    if "DATAPATH" in os.environ:
        structure = {subdir: os.environ["DATAPATH"] +structure[subdir] for subdir in structure}
    return structure

def load_parameters(model_name,exp):

    dir_structure = create_dir_structure(model_name,exp)
    saved_config = path.join(dir_structure["config"], "config.yaml")

    if path.isfile(saved_config):
        with open(saved_config, "r") as f:
            cdict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")


    return cdict, dir_structure,

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str,
                        default="fixed_length_model",
                        help="Source directory.")
    parser.add_argument("--gpu", type=int, required=True, help="The target device.")
    parser.add_argument("--mode", default="metrics", type=str, choices=["metrics", "fvd"],
                        help="The mode in which the test-method should be executed.")
    parser.add_argument("--metrics_on_patches", default=False, action="store_true",
                        help="Whether to run evaluation on patches (if available or not).")
    parser.add_argument("--best_ckpt", default=False, action="store_true",
                        help="Whether to use the best ckpt as measured by LPIPS (otherwise, latest_ckpt is used)")
    args = parser.parse_args()

    with open("config/model_names.txt", "r") as f:
        model_names = f.readlines()

    model_names = [m for m in model_names if not m.startswith("#")]

    gpu = args.gpu

    for model in model_names:
        model = model.rstrip()
        print(f"Evaluate model : {model}")

        cdict, dirs = load_parameters(model,args.experiment)

        cdict["testing"].update({"mode":args.mode})
        cdict["general"]["mode"] = "test"
        cdict["testing"].update({"best_ckpt": args.best_ckpt})
        cdict["testing"]["metrics_on_patches"] = args.metrics_on_patches
        cdict["general"]["restart"] = True
        experiment = select_experiment(cdict, dirs, args.gpu)
        try:
            experiment.test()
        except FileNotFoundError as e:
            print(e)
