from utils.general import get_logger
import os
from os import path
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str,
                        default="/export/scratch/ablattma/visual_poking/fixed_length_model/generated",
                        help="Source directory.")
    parser.add_argument("--gpu", type=int, required=True, help="The target device.")

    args = parser.parse_args()


    with open("config/model_names.txt", "r") as f:
        model_names = f.readlines()
    model_names = [m for m in model_names if not m.startswith("#")]
    logger = get_logger("eval-models")

    base_path = args.base

    if "DATAPATH" in os.environ:
        base_path = os.environ['DATAPATH'] + base_path

    logger.info(f'Base path is "{base_path}"')

    gpu = args.gpu

    for n in model_names:
        n = n.rstrip()
        logger.info(f"Compute fvd for model {n}")
        filepath = path.join(base_path, n, "samples_fvd")

        if not any(map(lambda x: x.endswith("npy"),os.listdir(filepath))):
            logger.info("no samples were found...skipping")

        try:
            test_cmd = f"python -m utils.metric_fvd --source {filepath} --gpu {gpu}"
            os.system(test_cmd)
        except Exception as e:
            logger.error(e)
            logger.info("next model")
            continue

    logger.info("finished")



