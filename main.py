import argparse
from os import path, makedirs
from experiments import select_experiment
import torch
import yaml
import os

def create_dir_structure(config):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["base_dir"],config["experiment"],subdir,config["project_name"]) for subdir in subdirs}
    if "DATAPATH" in os.environ:
        structure = {subdir: path.join(os.environ["DATAPATH"],structure[subdir]) for subdir in structure}
    return structure

def load_parameters(config_name, restart,debug,project_name):
    with open(config_name,"r") as f:
        cdict = yaml.load(f,Loader=yaml.FullLoader)
    if debug:
        cdict['general']['project_name'] = 'debug'
    else:
        cdict['general']['project_name'] = project_name



    dir_structure = create_dir_structure(cdict["general"])
    saved_config = path.join(dir_structure["config"], "config.yaml")
    if restart:
        if path.isfile(saved_config):
            with open(saved_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    else:
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        if path.isfile(saved_config) and not debug:
            print(f"\033[93m" + "WARNING: Model has been started somewhen earlier: Resume training (y/n)?" + "\033[0m")
            while True:
                answer = input()
                if answer == "y" or answer == "yes":
                    with open(saved_config,"r") as f:
                        cdict = yaml.load(f, Loader=yaml.FullLoader)

                    restart = True
                    break
                elif answer == "n" or answer == "no":
                    with open(saved_config, "w") as f:
                        yaml.dump(cdict, f, default_flow_style=False)
                    break
                else:
                    print(f"\033[93m" + "Invalid answer! Try again!(y/n)" + "\033[0m")
        else:
            with open(saved_config, "w") as f:
                yaml.dump(cdict,f,default_flow_style=False)

    cdict['general']['debug'] = debug
    return cdict, dir_structure, restart


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/latent_flow_net.yaml",
                        help="Define config file")
    parser.add_argument('-p','--project_name',type=str,default='ii2v',help='unique name for the training run to be (re-)started.')
    parser.add_argument("-r","--restart", default=False,action="store_true",help="Whether training should be resumed.")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Whether training should be resumed.")
    parser.add_argument("--gpu",default=[0], type=int,
                        nargs="+",help="GPU to use.")
    parser.add_argument("-m","--mode",default="train",type=str,choices=["train","test"],help="Whether to start in train or infer mode?")
    parser.add_argument("--test_mode",default="metrics",type=str, choices=["noise_test","metrics","fvd",'diversity','render'], help="The mode in which the test-method should be executed.")
    parser.add_argument("--metrics_on_patches", default=False,action="store_true",help="Whether to run evaluation on patches (if available or not).")
    parser.add_argument("--best_ckpt", default=False, action="store_true",help="Whether to use the best ckpt as measured by LPIPS (otherwise, latest_ckpt is used)")

    args = parser.parse_args()

    config, structure, restart = load_parameters(args.config, args.restart or args.mode == "test",args.debug,args.project_name)
    config["general"]["restart"] = restart
    config["general"]["mode"] = args.mode
    # config["general"]["first_stage"] = args.first_stage

    if len(args.gpu) == 1:
        gpus = torch.device(
            f"cuda:{int(args.gpu[0])}"
            if torch.cuda.is_available() and int(args.gpu[0]) >= 0
            else "cpu"
        )
        torch.cuda.set_device(gpus)
    else:
        gpus = [int(id) for id in args.gpu]

    mode = config["general"]["mode"]
    config["testing"].update({"best_ckpt": args.best_ckpt})
    if mode == "test" and "testing" in config and "metrics_on_patches" in config["testing"]:
        config["testing"]["metrics_on_patches"] = args.metrics_on_patches

    experiment = select_experiment(config, structure, gpus)

    # start selected experiment

    if  mode == "train":
        experiment.train()
    elif mode == "test":
        config["testing"].update({"mode": args.test_mode})
        experiment.test()
    else:
        raise ValueError(f"\"mode\"-parameter should be either \"train\" or \"infer\" but is actually {mode}")