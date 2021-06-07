import os
import cv2
from copy import deepcopy
import argparse
import torch
import numpy as np
from os import path, makedirs, listdir
import pickle
from tqdm import tqdm
import imagesize
from glob import glob
from natsort import natsorted
import yaml
import multiprocessing as mp
from multiprocessing import Process
from functools import partial


h36m_aname2aid = {name: i for i, name in enumerate(["Directions","Discussion","Eating","Greeting","Phoning",
                                                    "Posing","Purchases","Sitting","SittingDown","Smoking",
                                                    "Photo","Waiting","Walking","WalkDog","WalkTogether"])}
h36m_aname2aid.update({"WalkingTogether": h36m_aname2aid["WalkTogether"]})
h36m_aname2aid.update({"WalkingDog": h36m_aname2aid["WalkDog"]})
h36m_aname2aid.update({"TakingPhoto": h36m_aname2aid["Photo"]})


def _do_parallel_data_prefetch(func, Q, data, idx):
    # create dummy dataset instance

    # run prefetching
    res = func(data)
    Q.put([idx, res])
    Q.put("Done")

def get_image(vidcap, frame_number,spatial_size=None):
    vidcap.set(1, frame_number)
    _, img = vidcap.read()
    if spatial_size is not None and spatial_size != img.shape[0]:
        img=cv2.resize(img,(spatial_size,spatial_size),interpolation=cv2.INTER_LINEAR)
    return img


def process_images(d_name,semaphore, args):
    from utils.flownet_loader import FlownetPipeline
    from utils.general import get_gpu_id_with_lowest_memory, get_logger
    target_gpus = None if len(args.target_gpus) == 0 else args.target_gpus
    gpu_index = get_gpu_id_with_lowest_memory(target_gpus=target_gpus)
    logger = get_logger(f"{d_name}-{gpu_index}")
    img_list = natsorted([n for n in glob(path.join(d_name, f"*.{args.image_format}")) if n.split("/")[-1].startswith(args.image_prefix)])

    #basedir_name = args.raw_dir.split("*", 1)[0] if "*" in args.raw_dir else "/" + "/".join(args.raw_dir.split("/")[:-1])
    basedir_name = "/".join(args.raw_dir.split("/")[:-1])

    # test_proc = deepcopy(args.processed_dir)
    # if deepcopy(basedir_name).replace("/", "") == test_proc.replace("/", ""):
    #     flows = natsorted([n for n in glob(path.join(d_name, f"prediction_*.npy"))])
    #     n_flow_per_img = int(args.flow_max / args.flow_delta)
    #
    #     if len(flows) > n_flow_per_img * float(len(img_list) - args.flow_max - 1) / args.frames_discr:
    #         logger.info(f"Skipping dir {d_name} as flows are already extracted")
    #         semaphore.release()
    #         return



    torch.cuda.set_device(gpu_index)

    extract_device = torch.device("cuda", gpu_index.index if isinstance(gpu_index, torch.device) else gpu_index)

    # load flownet
    pipeline = FlownetPipeline()
    flownet = pipeline.load_flownet(args, extract_device)

    # get images



    subdir_name = d_name.split(basedir_name)[-1]
    if subdir_name.startswith("/"):
        subdir_name =subdir_name[1:]
    # path for saving the images
    base_path = path.join(args.processed_dir, subdir_name)

    makedirs(base_path, exist_ok=True)
    logger.info(f"Basepath is {base_path}")

    delta = args.flow_delta
    diff = args.flow_max



    number_frames = len(img_list)

    # only required for splitting the images, as split throws error when separator is empty string
    split_prefix = args.image_prefix if args.image_prefix != "" else "_"

    if args.continuous:
        for img_p in img_list[::args.frames_discr]:
            actual_id = img_p.split(split_prefix)[-1].split(f".{args.image_format}")[0]
            n_digits = len(actual_id)
            first_id = int(actual_id)
            second_id = first_id + diff
            # resave images, if intended
            img = None
            if args.resave_imgs:
                img_target_file = path.join(base_path, f"frame_{actual_id}.png")
                if not path.exists(img_target_file):
                    img = cv2.imread(img_p)
                    if args.spatial_size is not None and args.spatial_size != img.shape[0]:
                        img_resized = cv2.resize(img, (args.spatial_size, args.spatial_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_resized = img

                    # save resized image but use original image as input to the flownet later
                    success = cv2.imwrite(img_target_file, img_resized)

                    if success:
                        logger.info(f'wrote img with shape {img_resized.shape} to "{img_target_file}".')

            # FLOW
            for d in range(0, diff, delta):
                if second_id - d < number_frames:
                    flow_target_file = path.join(
                        base_path, f"prediction_{first_id}_{second_id - d}"
                    )
                    if not os.path.exists(flow_target_file + ".npy"):
                        # predict and write flow prediction
                        img_p2 = path.join(d_name, f"{args.image_prefix}{str(second_id - d).zfill(n_digits)}.{args.image_format}")
                        if img is None:
                            img = cv2.imread(img_p)
                        img2 = cv2.imread(img_p2)

                        sample = pipeline.preprocess_image(img, img2, "BGR", spatial_size=args.input_size).to(
                            extract_device
                        )
                        prediction = (
                            pipeline.predict(flownet, sample[None], spatial_size=args.spatial_size)
                                .cpu()
                                .detach()
                                .numpy()
                        )
                        np.save(flow_target_file, prediction)

                        logger.info(
                            f'wrote flow map with shape {prediction.shape} to "{flow_target_file}".')
    else:
        for img_count,img_p in enumerate(img_list):
            actual_id = img_p.split(split_prefix)[-1].split(f".{args.image_format}")[0]
            # n_digits = len(actual_id)
            # first_id = int(actual_id)

            img = None
            if args.resave_imgs:
                img_target_file = path.join(base_path, f"frame_{img_count}.png")
                if not path.exists(img_target_file):
                    img = cv2.imread(img_p)
                    if args.spatial_size is not None and args.spatial_size != img.shape[0]:
                        img_resized = cv2.resize(img, (args.spatial_size, args.spatial_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_resized = img

                    # save resized image but use original image as input to the flownet later
                    success = cv2.imwrite(img_target_file, img_resized)

                    if success:
                        logger.info(f'wrote img with shape {img_resized.shape} to "{img_target_file}".')

            # FLOW
            if img_count < len(img_list) - diff - 1:
                target_imgs = img_list[img_count+delta:img_count+diff+1:delta]
                for t_c,t_img in enumerate(target_imgs):
                    flow_target_file = path.join(
                        base_path, f"prediction_{img_count}_{img_count + (t_c+1) * delta}"
                    )
                    if not os.path.exists(flow_target_file + ".npy"):
                        # predict and write flow prediction
                        img_p2 = path.join(d_name, t_img)
                        if img is None:
                            img = cv2.imread(img_p)
                        img2 = cv2.imread(img_p2)

                        sample = pipeline.preprocess_image(img, img2, "BGR", spatial_size=args.input_size).to(
                            extract_device
                        )
                        prediction = (
                            pipeline.predict(flownet, sample[None], spatial_size=args.spatial_size)
                                .cpu()
                                .detach()
                                .numpy()
                        )
                        np.save(flow_target_file, prediction)

                        logger.info(
                            f'wrote flow map with shape {prediction.shape} to "{flow_target_file}".')




    semaphore.release()


def process_video(f_name, args):
    from utils.flownet_loader import FlownetPipeline
    from utils.general import get_gpu_id_with_lowest_memory, get_logger


    target_gpus = None if len(args.target_gpus) == 0 else args.target_gpus
    gpu_index = get_gpu_id_with_lowest_memory(target_gpus=target_gpus)
    torch.cuda.set_device(gpu_index)

    #f_name = vid_path.split(vid_path)[-1]

    logger = get_logger(f"{gpu_index}")

    extract_device = torch.device("cuda", gpu_index.index if isinstance(gpu_index,torch.device) else gpu_index)

    # load flownet
    pipeline = FlownetPipeline()
    flownet = pipeline.load_flownet(args, extract_device)

    # open video
    base_raw_dir = args.raw_dir.split("*")[0]

    if not isinstance(f_name,list):
        f_name = [f_name]

    logger.info(f"Iterating over {len(f_name)} files...")
    for fn in tqdm(f_name,):
        vid_path = path.join(base_raw_dir, fn)
        # vid_path = f"Code/input/train_data/movies/{fn}"
        vidcap = cv2.VideoCapture()
        vidcap.open(vid_path)
        counter = 0
        while not vidcap.isOpened():
            counter += 1
            time.sleep(1)
            if counter > 10:
                raise Exception("Could not open movie")

        # get some metadata
        number_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #upright = height > width

        # create target path if not existent

        base_path = path.join(args.processed_dir, fn.split(".")[0].replace("1024",str(args.spatial_size)))
        # base_path = f"Code/input/train_data/images/{f_name.split('.')[0]}/"
        makedirs(base_path, exist_ok=True)

        delta = args.flow_delta
        diff = args.flow_max

        # begin extraction
        for frame_number in range(0, number_frames):
            # check for existence
            first_fidx, second_fidx = frame_number, frame_number + diff
            image_target_file = path.join(base_path, f"frame_{frame_number}.png")
            # image_target_file = f"{base_path}frame_{frame_number}.png"
            # FRAME
            if not path.exists(image_target_file):
                # write frame itself
                img = get_image(vidcap, frame_number)
                if img is None:
                    continue
                # if upright:
                #     img = cv2.transpose(img)
                try:
                    if args.spatial_size is None:
                        success = cv2.imwrite(image_target_file, img)
                    else:
                        img_res = cv2.resize(img,(args.spatial_size,args.spatial_size), interpolation=cv2.INTER_LINEAR)
                        success = cv2.imwrite(image_target_file,img_res)
                except cv2.error as e:
                    print(e)
                    continue
                except Exception as ex:
                    print(ex)
                    continue

                # if success:
                #     logger.info(f'wrote img with shape {img.shape} to "{image_target_file}".')
            # FLOW
            for d in range(0, diff, delta):
                if second_fidx - d < number_frames:
                    flow_target_file = path.join(
                        base_path, f"prediction_{first_fidx}_{second_fidx-d}.flow"
                    )
                    if not os.path.exists(flow_target_file + ".npy"):
                        # predict and write flow prediction
                        img, img2 = (
                            get_image(vidcap, first_fidx),
                            get_image(vidcap, second_fidx - d),
                        )
                        # if upright:
                        #     img, img2 = cv2.transpose(img), cv2.transpose(img2)
                        sample = pipeline.preprocess_image(img, img2, "BGR",spatial_size=args.input_size).to(
                            extract_device
                        )
                        prediction = (
                            pipeline.predict(flownet, sample[None],spatial_size=args.spatial_size)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        np.save(flow_target_file, prediction)

        logger.info(
            f'Finish processing video sequence "{fn}".')

    return "Finish"

def extract(args):


    if args.process_vids:

        base_dir = args.raw_dir.split("*")[0]
        data_names = [p.split(base_dir)[-1] for p in glob(args.raw_dir) if p.endswith(args.video_format)]

        if args.restart:
            existing_ps = [p for p in data_names if path.isdir(path.join(args.processed_dir,p.split(".")[0]))]
            data_names = [p for p in data_names if not path.isdir(path.join(args.processed_dir,p.split(".")[0]))]


        if any(map(lambda x: "vegetation" in x,args.processed_dir.split("/"))):
            target_name = "train" if "TRAIN" in args.raw_dir.split("/") else "test"
            with open(path.join(args.processed_dir,f"{target_name}.yaml"),"r") as f:
                target_vids = yaml.load(f,Loader=yaml.FullLoader)

            target_vids = [f"Stochastic_motion_{key}_c{val}.{args.video_format}" for key in target_vids for val in target_vids[key]]
            data_names = target_vids

            args.processed_dir = path.join(args.processed_dir,target_name)

            makedirs(args.processed_dir, exist_ok=True)

            data_names = [v for v in data_names if v.split(".")[0] not in listdir(args.processed_dir)]

        fn_extract = partial(process_video, args=args)

        Q = mp.Queue(1000)
        step = (
            int(len(data_names) / args.num_workers + 1)
            if len(data_names) % args.num_workers != 0
            else int(len(data_names) / args.num_workers)
        )
        arguments = [
            [fn_extract, Q, part, i]
            for i, part in enumerate(
                [data_names[i: i + step] for i in range(0, len(data_names), step)]
            )
        ]
        processes = []
        for i in range(args.num_workers):
            p = Process(target=_do_parallel_data_prefetch, args=arguments[i])
            processes += [p]

        start = time.time()
        gather_res = [[] for _ in range(args.num_workers)]
        try:
            for p in processes:
                p.start()

            k = 0
            while k < args.num_workers:
                # get result
                res = Q.get()
                if res == "Done":
                    k += 1
                else:
                    gather_res[res[0]] = res[1]

        except Exception as e:
            print("Exception: ", e)
            for p in processes:
                p.terminate()

            raise e
        finally:
            for p in processes:
                p.join()
            print(f"Prefetching complete. [{time.time() - start} sec.]")

    else:
        data_names = [p for p in glob(args.raw_dir) if path.isdir(p)]

        if args.filter_with_target:
            target_names = [f.split("/")[-2:] for f in glob(path.join(args.processed_dir,"*","*"))]


            tnames1 = [f[0] for f in target_names]
            tnames2 = [f[1] for f in target_names]

            data_names = [d for d in data_names if not (d.split("/")[-2] in tnames1 and d.split("/")[-1] in tnames2)]

        fn_extract = process_images

        makedirs(args.processed_dir, exist_ok=True)
        if args.use_mp:
            # fixme only hacked for human3.6M
            basedir_name = "/".join(args.raw_dir.split("/")[:-1])

            test_proc = deepcopy(args.processed_dir)

            for name in data_names:
                img_list = natsorted([n for n in glob(path.join(name, f"*.{args.image_format}")) if n.split("/")[-1].startswith(args.image_prefix)])
                if deepcopy(basedir_name).replace("/", "") == test_proc.replace("/", ""):
                    flows = natsorted([n for n in glob(path.join(name, f"prediction_*.npy"))])
                    n_flow_per_img = int(args.flow_max / args.flow_delta)

                    if len(flows) > n_flow_per_img * float(len(img_list) - args.flow_max - 1) / args.frames_discr:
                        print(f"Skipping dir {name} as flows are already extracted")
                        continue


                p = Process(target=fn_extract, args=(name, semaphore, args))
                semaphore.acquire()
                p.start()
                pool.append(p)
                time.sleep(5)

            for th in pool:
                th.join()
        else:
            for name in data_names:
                fn_extract(name,None,args)

def prepare(args):
    logger = get_logger("dataset_preparation")


    datadict = {
        "img_path": [],
        "flow_paths": [],
        "fid": [],
        "vid": [],
        "img_size": [],
        "flow_size": [],
        "object_id":[],
        "max_fid": []
    }
    if "iPER" in args.processed_dir.split("/") or "human36m" in args.processed_dir.split("/") or \
            "human3.6M" in args.processed_dir.split("/") :
        datadict.update({"action_id": [], "actor_id": []})

    train_test_split = "human3.6M" in args.processed_dir.split("/")  or\
                       "vegetation_new" in args.processed_dir.split("/") or \
                       "bair" in args.processed_dir.split("/") or "taichi" in args.processed_dir.split("/")

    if train_test_split:
        datadict.update({"train": []})
        if "taichi" in args.processed_dir.split("/"):
            oname2oid = {}

    logger.info(f'Metafile is stored as "{args.meta_file_name}.p".')
    logger.info(f"args.check_imgs is {args.check_imgs}")
    max_flow_length = int(args.flow_max / args.flow_delta)
    n_dig = len(str(args.flow_max))

    if args.process_vids:
        if train_test_split:
            videos = [d for d in glob(path.join(args.processed_dir, "*", "*")) if path.isdir(d)]
        else:
            videos = [d for d in glob(path.join(args.processed_dir, "*")) if path.isdir(d)]

        videos = natsorted(videos)

        actual_oid = 0
        for vid, vid_name in enumerate(videos):

            images = glob(path.join(vid_name, "*.png"))
            images = natsorted(images)

            actor_id = action_id = train = None
            if "plants" in args.processed_dir.split("/"):
                object_id = int(vid_name.split("/")[-1].split("_")[1])
            elif "iPER" in args.processed_dir.split("/"):
                object_id = 100 * int(vid_name.split("/")[-1].split("_")[0]) + int(vid_name.split("/")[-1].split("_")[1])
                actor_id = int(vid_name.split("/")[-1].split("_")[0])
                action_id = int(vid_name.split("/")[-1].split("_")[-1])
            elif train_test_split:
                train = "train" == vid_name.split("/")[-2]
                msg = "train" if train else "test"
                print(f"Video in {msg}-split")
                if "taichi" in args.processed_dir.split("/"):
                    obj_name = vid_name.split("/")[-1].split("#")[0]
                    if obj_name in oname2oid.keys():
                        object_id = oname2oid[obj_name]
                    else:
                        object_id = actual_oid
                        oname2oid.update({obj_name: actual_oid})
                        actual_oid += 1
                elif "bair" in args.processed_dir.split("/"):
                    object_id = 0
                elif "vegetation_new" in args.processed_dir.split("/"):
                    object_id = actual_oid
                    actual_oid+=1
                else:
                    uname = vid_name.split("/")[-1].split("_")
                    object_id = 10 * int(uname[-1][1:]) + 10000 * (int(uname[-2][-1])) + int(train)
            else:
                raise ValueError("invalid dataset....")

            max_flow_id = [len(images) - flow_step -1 for flow_step in range(args.flow_delta,args.flow_max+1, args.flow_delta)]
            for i, img_path in enumerate(
                    tqdm(
                        images,
                        desc=f'Extracting meta information of video "{vid_name.split("/")[-1]}"',
                    )
            ):
                fid = int(img_path.split("_")[-1].split(".")[0])
                #search_pattern = f'[{",".join([str(fid + n) for n in range(args.flow_delta,args.flow_max + 1, args.flow_delta)])}]'

                flows = natsorted([s for s in glob(path.join(vid_name, f"prediction_{fid}_*.npy"))
                                   if (int(s.split("_")[-1].split(".")[0]) - int(s.split("_")[-2])) % args.flow_delta == 0 and
                                   int(s.split("_")[-1].split(".")[0]) - int(s.split("_")[-2]) <= args.flow_max])

                # make relative paths
                img_path_rel = img_path.split(args.processed_dir)[1]
                flows_rel = [f.split(args.processed_dir)[1] for f in flows]
                # filter flows
                flows_rel = [f for f in flows_rel if (int(f.split("/")[-1].split(".")[0].split("_")[-1]) - int(f.split("/")[-1].split(".")[0].split("_")[-2])) <= args.flow_max]

                if len(flows_rel) < max_flow_length:
                    diff = max_flow_length-len(flows_rel)
                    [flows_rel.insert(len(flows_rel),last_flow_paths[len(flows_rel)]) for _ in range(diff)]

                # if len(flows_rel) < max_flow_length and len(flows) > 0:
                #     n_append = int(max_flow_length // len(flows_rel))
                #     flows_rel = flows_rel + n_append * flows_rel
                #     flows_rel = flows_rel[:max_flow_length]
                # elif len(flows_rel) == 0:
                #     flows_rel = max_flow_length * [path.join(args.processed_dir,"end")]

                if args.check_imgs:
                    w_img, h_img = imagesize.get(img_path)
                    if len(flows) > 0:
                        h_f, w_f = np.load(flows[0]).shape[:2]
                    else:
                        h_f = w_f = None
                else:
                    w_img = args.spatial_size
                    h_img = args.spatial_size
                    if len(flows) > 0:
                        w_f = args.spatial_size
                        h_f = args.spatial_size
                    else:
                        h_f = w_f = None

                assert len(flows_rel) == max_flow_length
                datadict["img_path"].append(img_path_rel)
                datadict["flow_paths"].append(flows_rel)
                datadict["fid"].append(fid)
                datadict["vid"].append(vid)
                # image size compliant with numpy and torch
                datadict["img_size"].append((h_img, w_img))
                datadict["flow_size"].append((h_f, w_f))
                datadict["object_id"].append(object_id)
                datadict["max_fid"].append(max_flow_id)
                if action_id is not None:
                    datadict["action_id"].append(action_id)
                if actor_id is not None:
                    datadict["actor_id"].append(actor_id)
                if train is not None:
                    datadict["train"].append(train)

                last_flow_paths = flows_rel

    else:

        basedir_name = args.raw_dir.split("*", 1)[0] if "*" in args.raw_dir else "/".join(args.raw_dir.split("/")[:-1])
        split_prefix = args.image_prefix if args.image_prefix != "" else "_"

        if train_test_split:
            datadict.update({"train":[]})

        if args.resave_imgs:
            subdirs = glob(args.raw_dir)[0].split(basedir_name)[-1].split("/")
            n_subdirs = len(subdirs)
            data_dirs = [d for d in glob(path.join(args.processed_dir, *["*" for n in range(n_subdirs)])) if path.isdir(d)]

            if "human36m" in args.processed_dir.split("/") or "human3.6M" in args.processed_dir.split("/") or "plants" in args.processed_dir.split("/") or "iPER" in args.processed_dir.split("/"):
                del datadict["max_fid"]
            fid = 0
            for vid, dir_name in enumerate(data_dirs):
                images = natsorted(glob(path.join(dir_name, "*.png")))
                action_id = actor_id = None

                # [1:] to remove leading "/"
                dir_name_rel = dir_name.split(args.processed_dir)[-1]

                #logger.info(f"Processing images and flows in directory \"{dir_name_rel}\".")

                if dir_name_rel.startswith("/"):
                    dir_name_rel = dir_name_rel[1:]

                if "human3.6M" in args.processed_dir.split("/")  and dir_name_rel.split("/")[1].startswith("-"):
                    logger.info(f"Skipping {dir_name_rel}")
                    continue



                if "human36m" in args.processed_dir.split("/"):
                    object_id = int(dir_name_rel.split("/")[0][1:])
                    actor_id = object_id
                    action_id = h36m_aname2aid[dir_name_rel.split("/")[1][:-2]]
                elif "human3.6M" in args.processed_dir.split("/"):
                    object_id = int(dir_name_rel.split("/")[1].split("-")[0][1:])
                    actor_id = object_id
                    astring = dir_name_rel.split("/")[1].split("-")[1].split(".")[0]
                    if "_" in astring:
                        astring=astring.split("_")[0]
                    action_id = h36m_aname2aid[astring]
                elif "plants" in args.processed_dir.split("/"):
                    object_id = int(dir_name_rel.split("/")[-1].split("_")[1])
                elif "iPER" in args.processed_dir.split("/"):
                    object_id = 100 * int(dir_name.split("/")[-1].split("_")[0]) + int(dir_name.split("/")[-1].split("_")[1])
                    actor_id = int(dir_name.split("/")[-1].split("_")[0])
                    action_id = int(dir_name.split("/")[-1].split("_")[-1])
                else:
                    raise ValueError(f"Invalid image dataset with processed_dir \"{args.processed_dir}\".")

                for i, img_p in enumerate(tqdm(images[::args.frames_discr],desc=f"Processing images and flows in directory \"{dir_name}\".")):
                    img_filename = img_p.split("/")[-1]
                    target_fid = int(img_filename.split(args.image_prefix)[-1].split(".")[0])
                    img_name_rel = path.join(dir_name_rel,img_filename)
                    if img_name_rel.startswith("/"):
                        img_name_rel = img_name_rel[1:]

                    # fid = int(img_filename.split("frame_")[-1].split(".")[0])
                    flow_names = natsorted(glob(path.join(dir_name,f"prediction_{target_fid}_*.npy")))
                    flow_names_rel = [fname.split(args.processed_dir)[-1] for fname in flow_names]
                    flow_names_rel = [fname[1:] if fname.startswith("/") else fname for fname in flow_names_rel ]

                    if len(flow_names_rel) < max_flow_length:
                        continue

                    h = w = args.spatial_size

                    if train_test_split:
                        train = "train" in dir_name_rel
                        datadict["train"].append(train)

                    datadict["img_path"].append(img_name_rel)
                    datadict["flow_paths"].append(flow_names_rel)
                    datadict["fid"].append(fid)
                    datadict["vid"].append(vid)
                    # image size compliant with numpy and torch
                    datadict["img_size"].append((1024, 1024))
                    datadict["flow_size"].append((h, w))
                    datadict["object_id"].append(object_id)
                    if action_id is not None:
                        datadict["action_id"].append(action_id)
                    if actor_id is not None:
                        datadict["actor_id"].append(actor_id)

                    fid += 1

            # ensure that fids are 0-based
            vid_arr = np.asarray(datadict["vid"])
            fid_arr = np.asarray(datadict["fid"])
            # sorting vids in unique does not affect order since vids are sorted anyways
            for vid in np.unique(vid_arr):
                min_fid = np.amin(fid_arr[vid_arr==vid])
                fid_arr[vid_arr==vid]-= min_fid

            datadict["fid"] = fid_arr.tolist()


        else:

            actual_oid = 0
            target_lags = None
            img_p_rel = False
            if train_test_split:
                datadict.update({"train": []})
                if "taichi" in args.processed_dir.split("/"):
                    oname2oid = {}
                    del datadict["max_fid"]
                    img_p_rel = True
                    target_lags = [10,20]
                else:
                    raise NotImplementedError()

            for vid,img_dir in enumerate(glob(args.raw_dir)):

                action_id = actor_id = train = None
                if "human36m" in args.processed_dir.split("/"):
                    object_id = int(img_dir.split(basedir_name)[-1].split("/")[0][1:])
                    actor_id = object_id
                    action_id = h36m_aname2aid[img_dir.split(basedir_name)[-1].split("/")[1][:-2]]
                elif "sky" in args.processed_dir.split("/"):
                    object_id = vid
                elif train_test_split:
                    train = "train" == img_dir.split("/")[-2]
                    msg = "train" if train else "test"
                    print(f"Video in {msg}-split")
                    if "taichi" in args.processed_dir.split("/"):
                        obj_name = img_dir.split("/")[-1].split("#")[0]
                        if obj_name in oname2oid.keys():
                            object_id = oname2oid[obj_name]
                        else:
                            object_id = actual_oid
                            oname2oid.update({obj_name: actual_oid})
                            actual_oid += 1
                else:

                    raise ValueError(f"Invalid image dataset with processed_dir \"{args.processed_dir}\".")
                subdir_name = img_dir.split(basedir_name)[-1]
                flow_dir = path.join(args.processed_dir,subdir_name)

                for fid,img_p in enumerate(tqdm(natsorted([n for n in glob(path.join(img_dir,f"*.{args.image_format}")) if n.split("/")[-1].startswith(args.image_prefix)]),
                                                desc=f"processing images in directory \"{img_dir}\"")):

                    actual_id = img_p.split(split_prefix)[-1].split(f".{args.image_format}")[0]
                    first_id = int(actual_id)
                    flow_ps_rel = [p.split(args.processed_dir)[-1] for p in natsorted(glob(path.join(flow_dir, f"prediction_{first_id}_*.npy")))]
                    if target_lags is not None:
                        flow_ps_rel = [p for p in flow_ps_rel if int(p.split("_")[-1].split(".")[0])-int(p.split("_")[-2]) in target_lags]
                        if len(flow_ps_rel) != len(target_lags):
                            continue

                    if img_p_rel:
                        img_path_rel = img_p.split(args.processed_dir)[-1]
                    else:
                        img_path_rel = img_p

                    if len(flow_ps_rel) <max_flow_length:
                        continue

                    w_img, h_img = imagesize.get(img_p)
                    img_p = img_path_rel
                    if len(flow_ps_rel) > 0:
                        w_f = args.spatial_size
                        h_f = args.spatial_size
                    else:
                        h_f = w_f = None

                    datadict["img_path"].append(img_p)
                    datadict["flow_paths"].append(flow_ps_rel)
                    datadict["fid"].append(fid)
                    datadict["vid"].append(vid)
                    # image size compliant with numpy and torch
                    datadict["img_size"].append((h_img, w_img))
                    datadict["flow_size"].append((h_f, w_f))
                    datadict["object_id"].append(object_id)
                    if action_id is not None:
                        datadict["action_id"].append(action_id)
                    if actor_id is not None:
                        datadict["actor_id"].append(actor_id)
                    if train is not None:
                        datadict["train"].append(train)

    logger.info(f'Prepared dataset consists of {len(datadict["img_path"])} samples.')

    # Store data (serialize)
    save_path = path.join(
        args.processed_dir, f"{args.meta_file_name}.p"
    )
    with open(save_path, "wb") as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

    import time
    from utils.general import get_logger




    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--process_vids","-pv",default=False,action="store_true",help="Whether to process videos (true) or images (false).")
    parser.add_argument("--rgb_max", type=float, default=1.0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).",
    )
    parser.add_argument(
        "--fp16_scale",
        type=float,
        default=1024.0,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--raw_dir",
        "-v",
        type=str,
        default="/export/data/ablattma/Datasets/plants/cropped/",
    )
    parser.add_argument(
        "--processed_dir",
        "-p",
        type=str,
        default="/export/scratch/ablattma/Datasets/plants/processed_256/",
    )
    parser.add_argument(
        "--flow_delta",
        "-fd",
        type=int,
        default=5,
        help="The number of frames between two subsequently extracted flows.",
    )
    parser.add_argument("--flow_max", "-fm", type=int, default=30)
    parser.add_argument(
        "--mode", type=str, choices=["all", "extract", "prepare"], default="all"
    )
    parser.add_argument(
        "--check_imgs",
        "-ci",
        type=bool,
        default=False,
        help="Whether to check the images for their size or not (more time consumpting, if enabled).",
    )
    parser.add_argument("--meta_file_name","-mfn",type=str,default="meta_data", help="The name for the pickle file, where the meta data is stored (without ending).")
    parser.add_argument("--video_format", "-vf", type=str, default="mkv", choices=["mkv","mp4"],help="Format of the input videos to the pipeline.")
    parser.add_argument("--spatial_size", "-s",type=int, default=256,help="The desired spatial_size of the output.")
    parser.add_argument("--image_format","-if",type=str, default="png", choices=["png","jpg"],help="Format of the input images to the pipeline.")
    parser.add_argument("--image_prefix", "-ip", type=str,default = "_", help="The prefix to the images.")
    parser.add_argument("--input_size", "-is", type=int, default=1024, help="The input size for the flownet (images are resized to this size, if not divisible by 64.")
    parser.add_argument("--resave_imgs","-ri",action="store_true",default=False,help="Whether to re-save the resized images when processing images instead of videos (default: False).")
    parser.add_argument("--frames_discr", "-fdi", type=int, default=1, help="The discretization step for images to take from the source img dir.")
    parser.add_argument("--target_gpus", default=[], type=int,
                        nargs="+", help="GPU's to use.")
    parser.add_argument("--filter_with_target", "-ft", action="store_true", default=False, help="Whether to filter the processed data with target names from the args.processed_dir (default: False).")
    parser.add_argument("--restart", "-r", action="store_true", default=False, help="Whether script is restarted due to a former crash.")
    parser.add_argument("--num_workers","-nw",type=int, default=1,help="The number of parallel processes that will be started for etxracting the data.")
    parser.add_argument("--use_mp",action="store_true", default=False, help="Whether to use mp or not.")
    parser.add_argument("--continuous",action="store_true", default=False, help="Whether or not img ids are assumed to be in continuous.")
    args = parser.parse_args()

    pool = []
    torch.multiprocessing.set_start_method("spawn")
    if not args.process_vids:
        semaphore = mp.Semaphore(args.num_workers)

    if args.mode == "extract":
        extract(args)
    elif args.mode == "prepare":  # in this case, it is prepare
        prepare(args)
    else:
        extract(args)
        prepare(args)
