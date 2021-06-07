import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import cv2
import math
import imutils
import matplotlib.pyplot as plt
import wandb
from os import path
import math

def make_flow_grid(src, poke, pred, tgt, n_logged, flow=None):
    """

    :param src:
    :param poke:
    :param pred:
    :param tgt:
    :param n_logged:
    :return:
    """
    src = ((src.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    # poke = poke.permute(0, 2, 3, 1).cpu().numpy()[:n_logged]
    # poke -= poke.min()
    # poke /= poke.max()
    # poke = (poke * 255.0).astype(np.uint8)
    # poke = np.concatenate([poke, np.expand_dims(np.zeros_like(poke).sum(-1), axis=-1)], axis=-1).astype(np.uint8)
    poke = vis_flow(poke[:n_logged])
    poke = np.concatenate(poke,axis=1)
    # if prediction is image, just take the premuted image
    if pred.shape[1] == 3:
        pred = ((pred.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(
            np.uint8)[:n_logged]
    else:
        # if prediction is flow, you need treat it like that
        pred = pred.permute(0, 2, 3, 1).cpu().numpy()[:n_logged]
        pred -= pred.min()
        pred /= pred.max()
        pred = (pred * 255.0).astype(np.uint8)
        pred = np.concatenate([pred, np.expand_dims(np.zeros_like(pred).sum(-1), axis=-1)], axis=-1).astype(np.uint8)
    tgt = ((tgt.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(
        np.uint8)[:n_logged]


    tgt_gr = [cv2.cvtColor(t,cv2.COLOR_RGB2GRAY) for t in tgt]
    pred_gr = [cv2.cvtColor(t,cv2.COLOR_RGB2GRAY) for t in pred]
    ssim_imgs = [ssim(rimg, fimg, multichannel=True, data_range=255, gaussian_weights=True, use_sample_covariance=False, full=True)[1] for rimg, fimg in zip(tgt_gr, pred_gr)]
    additional = [np.concatenate([cv2.cvtColor((s * 255.).astype(np.uint8),cv2.COLOR_GRAY2RGB) for s in ssim_imgs],axis=1)]



    if flow is not None:
        # if provided, use additional flow information (in case of poking, that's the entire flow src --> tgt
        # add = flow.permute(0, 2, 3, 1).cpu().numpy()[:n_logged]
        # add -= add.min()
        # add /= add.max()
        # add = (add * 255.0).astype(np.uint8)
        # add = np.concatenate([add, np.expand_dims(np.zeros_like(add).sum(-1), axis=-1)], axis=-1).astype(np.uint8)
        # additional = additional + [np.concatenate([a for a in add], axis=1)]
        add = vis_flow(flow[:n_logged])
        add = np.concatenate(add,axis=1)
        additional = additional + [add]
        # compute ssim_img in grayscale

    src = np.concatenate([s for s in src], axis=1)
    #poke = np.concatenate([f for f in poke], axis=1)
    pred = np.concatenate([p for p in pred], axis=1)
    tgt = np.concatenate([t for t in tgt], axis=1)
    grid = np.concatenate([src,poke,pred,tgt,*additional],axis=0)

    return grid

def vis_flow(flow_map, normalize=False):
    if isinstance(flow_map,torch.Tensor):
        flow_map = flow_map.cpu().numpy()
    flows_vis = []
    for flow in flow_map:
        hsv = np.zeros((*flow.shape[1:],3),dtype=np.uint8)
        hsv[...,1] = 255
        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        # since 360 is not valid for uint8, 180° corresponds to 360° for opencv hsv representation. Therefore, we're dividing the angle by 2 after conversion to degrees
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)

        as_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        if normalize:
            as_rgb = as_rgb.astype(np.float) - as_rgb.min(axis=(0,1),keepdims=True)
            as_rgb = (as_rgb / as_rgb.max(axis=(0,1),keepdims=True)*255.).astype(np.uint8)
        flows_vis.append(as_rgb)

    return flows_vis

def vis_flow_dense(flow_map,**kwargs):
    if isinstance(flow_map,torch.Tensor):
        flow_map = flow_map.cpu().numpy()
    flows_vis = []
    for flow in flow_map:
        h, w = flow.shape[1:]
        fx, fy = flow[0], flow[1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(v,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flows_vis.append(bgr)
    return flows_vis

def make_trf_video(img1,img2,v12,v21,poke,n_logged,logwandb=True,length_divisor=5):
    """

    :param img1:
    :param img2:
    :param v12:
    :param v21:
    :param poke:
    :param n_logged:
    :param lomake_flow_grid()gwandb:
    :param length_divisor:
    :return:
    """
    seq_len = v12.shape[1]
    pokes = vis_flow(poke[:n_logged])

    img1 = ((img1.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    img2 = ((img2.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]

    img1_with_arrow = []
    img2_with_arrow = []
    eps = 1e-6
    for i, (poke_p, img1_i, img2_i) in enumerate(zip(poke[:n_logged], img1, img2)):
        poke_points = np.nonzero(pokes[i].any(-1) > 0)
        if poke_points[0].size == 0:
            img1_with_arrow.append(img1_i)
            img2_with_arrow.append(img2_i)
        else:
            min_y = np.amin(poke_points[0])
            max_y = np.amax(poke_points[0])
            min_x = np.amin(poke_points[1])
            max_x = np.amax(poke_points[1])
            # plot mean direction of flow in poke region
            avg_flow = np.mean(poke_p[:, min_y:max_y, min_x:max_x].cpu().numpy(), axis=(1, 2))
            arrow_dir = avg_flow / (np.linalg.norm(avg_flow) + eps) * (poke_p.shape[1] / length_divisor)
            if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                arrow_start = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                arrow_end = (arrow_start[0] + int(arrow_dir[0]), arrow_start[1] + int(arrow_dir[1]))

                img1_with_arrow.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img1_i), arrow_start, arrow_end, (255, 0, 0), max(int(img1_i.shape[0] / 64), 1))))
                img2_with_arrow.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img2_i), arrow_start, arrow_end, (255, 0, 0), max(int(img2_i.shape[0] / 64), 1))))
            else:
                img1.append(img1_i)
                img2.append(img2_i)

    vid_st1 = np.concatenate(img1_with_arrow, axis=1)
    vid_st2 = np.concatenate(img2_with_arrow, axis=1)

    vid_st1 = put_text_to_video_row(np.stack([vid_st1] * seq_len, axis=0), "Image 1", color=(255, 0, 0))
    vid_st2 = put_text_to_video_row(np.stack([vid_st2] * seq_len, axis=0), "Image 2", color=(255, 0, 0))

    v12 = ((v12.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    v12 = np.concatenate(list(v12), axis=2)
    v12 = put_text_to_video_row(v12, "Vid: FG1-BG2")

    v21 = ((v21.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    v21 = np.concatenate(list(v21), axis=2)
    v21 = put_text_to_video_row(v21, "Vid: FG2-BG1")

    full = np.concatenate([vid_st1, vid_st2, v12, v21], axis=1)
    if logwandb:
        full = np.moveaxis(full, [0, 1, 2, 3], [0, 2, 3, 1])

    return full

def draw_arrow(traj):

    arrow_imgs = []
    for c,t in enumerate(traj):
        active_points = np.nonzero(t.astype(np.uint8).any(0) > 0)

        img = np.zeros((*t.shape[1:],3),dtype=np.uint8)
        if active_points[0].size>0:
            for i in range(active_points[0].shape[0]):
                y =active_points[0][i]
                x = active_points[1][i]

                arrow_dir = t[:,y,x]
                if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                    arrow_start = (x, y)
                    arrow_end = (int(np.clip(x + int(arrow_dir[0]),0,img.shape[0])), int(np.clip(y + int(arrow_dir[1]),0,img.shape[0])))
                    img = cv2.arrowedLine(img, arrow_start, arrow_end, (255, 0, 0), max(int(traj.shape[1] / 64), 1))


        arrow_imgs.append(img)

    arrow_imgs = np.concatenate(arrow_imgs,axis=1)
    return arrow_imgs


def img_grid_ci(src,traj,pred,tgt,n_logged):
    src = ((src.permute(0, 2, 3, 1).cpu().numpy()) * 255.).astype(np.uint8)[:n_logged]

    # if prediction is image, just take the premuted image
    pred = ((pred.permute(0, 2, 3, 1).cpu().numpy()) * 255).astype(
        np.uint8)[:n_logged]

    tgt = ((tgt.permute(0, 2, 3, 1).cpu().numpy()) * 255).astype(
        np.uint8)[:n_logged]

    src = np.concatenate([s for s in src], axis=1)
    # poke = np.concatenate([f for f in poke], axis=1)
    pred = np.concatenate([p for p in pred], axis=1)
    tgt = np.concatenate([t for t in tgt], axis=1)

    arrows = draw_arrow(traj[:n_logged].cpu().numpy())
    grid = np.concatenate([src, arrows, pred, tgt], axis=0)

    return grid


def make_video_ci(src,traj,pred,tgt,n_logged,logwandb=True, display_frame_nr=True):
    seq_len = tgt.shape[1]

    srcs = np.concatenate([s for s in ((src.permute(0, 2, 3, 1).cpu().numpy()) * 255.).astype(np.uint8)[:n_logged]],axis=1)

    traj_vis = []
    for t in range(traj.shape[1]):

        arrows = draw_arrow(traj[:n_logged,t].cpu().numpy())
        traj_vis.append(arrows)

    traj_vis = np.stack(traj_vis,axis=0)
    traj_vis = put_text_to_video_row(traj_vis, "Flow Vectors", display_frame_nr=display_frame_nr)

    srcs = cv2.UMat.get(cv2.putText(cv2.UMat(srcs), f"Sequence length {seq_len}", (int(srcs.shape[1] // 3), int(srcs.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    float(srcs.shape[0] / 256), (255, 0, 0), int(srcs.shape[0] / 128)))
    srcs = np.stack([srcs] * seq_len, axis=0)
    srcs = put_text_to_video_row(srcs, "Input Image", display_frame_nr=display_frame_nr)

    pred = ((pred.permute(0, 1, 3, 4, 2).cpu().numpy()) * 255).astype(np.uint8)[:n_logged]
    pred = np.concatenate(list(pred), axis=2)
    pred = put_text_to_video_row(pred, "Predicted Video", display_frame_nr=display_frame_nr)

    tgt = ((tgt.permute(0, 1, 3, 4, 2).cpu().numpy()) * 255).astype(np.uint8)[:n_logged]
    tgt = np.concatenate(list(tgt), axis=2)
    tgt = put_text_to_video_row(tgt, "Groundtruth Video", display_frame_nr=display_frame_nr)

    full = np.concatenate([srcs, pred, tgt, traj_vis], axis=1)
    if logwandb:
        full = np.moveaxis(full, [0, 1, 2, 3], [0, 2, 3, 1])

    return full




def make_video(src,poke,pred,tgt,n_logged,flow=None,length_divisor=5,logwandb=True,flow_weights= None, display_frame_nr=False,invert_poke = False):
    """

    :param src: src image
    :param poke: poke, also input to the network
    :param pred: predicted video of the network
    :param tgt: target video the network was trained to reconstruct
    :param n_logged: numvber of logged examples
    :param flow: src flow from which the poke is originating
    :param length_divisor: divisor for the length of the arrow, that's drawn ti visualize the mean direction of the flow within the poke patch
    :param logwandb: whether the output video grid is intended to be logged with wandb or not (in this case the grid channels have to be changed)
    :param flow_weights: Optional weights for the flow which are also displayed if they are not None.
    :return:
    """
    seq_len = tgt.shape[1]

    src = ((src.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]

    pokes = vis_flow(poke[:n_logged])
    flows_vis = None
    if flow is not None:
        flows = vis_flow(flow[:n_logged])
        flows_with_rect = []
        for i,(poke_p,flow) in enumerate(zip(pokes,flows)):
            poke_points = np.nonzero(poke_p.any(-1) > 0)
            if poke_points[0].size == 0:
                flows_with_rect.append(np.zeros_like(flow))
            else:
                min_y = np.amin(poke_points[0])
                max_y = np.amax(poke_points[0])
                min_x = np.amin(poke_points[1])
                max_x = np.amax(poke_points[1])
                # draw rect
                flow_with_rect = cv2.rectangle(flow,(min_x,min_y),(max_x,max_y),(255,255,255),max(1,int(flow.shape[0]//64)))
                # flow_with_rect = cv2.UMat.get(cv2.putText(cv2.UMat(flow_with_rect), f"Flow Complete",(int(flow_with_rect.shape[1] // 3), int(5 * flow_with_rect.shape[0] / 6) ), cv2.FONT_HERSHEY_SIMPLEX,
                #                        float(flow_with_rect.shape[0] / 256), (255, 255, 255), int(flow_with_rect.shape[0] / 128)))

                flows_with_rect.append(flow_with_rect)

        flow_cat = np.concatenate(flows_with_rect,axis=1)

        flows_vis= [np.stack([flow_cat]*seq_len,axis=0)]
        flows_vis[0] = put_text_to_video_row(flows_vis[0], "Flow Complete", color=(255, 255, 255))

    if flow_weights is not None:
        flow_weights = flow_weights.cpu().numpy()
        heatmaps = []
        for i, weight in enumerate(flow_weights):
            weight_map = ((weight - weight.min()) / weight.max() * 255.).astype(np.uint8)
            heatmap = cv2.applyColorMap(weight_map, cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            heatmaps.append(heatmap)

        heatmaps = np.concatenate(heatmaps, axis=1)
        heatmaps = np.stack([heatmaps]*seq_len, axis=0)
        heatmaps = put_text_to_video_row(heatmaps, "Flow Weights", color=(255,255,255))
        if flows_vis is None:
            flows_vis = [heatmaps]
        else:
            flows_vis.insert(0,heatmaps)

    srcs_with_arrow = []
    pokes_with_arrow = []
    if invert_poke:
        srcs_with_arrow_inv = []
        pokes_with_arrow_inv = []
    eps = 1e-6
    for i, (poke_p,src_i) in enumerate(zip(poke[:n_logged],src)):
        poke_points = np.nonzero(pokes[i].any(-1) > 0)
        if poke_points[0].size==0:
            pokes_with_arrow.append(np.zeros_like(pokes[i]))
            srcs_with_arrow.append(src_i)
        else:
            min_y = np.amin(poke_points[0])
            max_y = np.amax(poke_points[0])
            min_x = np.amin(poke_points[1])
            max_x = np.amax(poke_points[1])
            # plot mean direction of flow in poke region
            avg_flow = np.mean(poke_p[:, min_y:max_y, min_x:max_x].cpu().numpy(), axis=(1, 2))
            arrow_dir = avg_flow / (np.linalg.norm(avg_flow) + eps) * (poke_p.shape[1] / length_divisor)
            if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                arrow_start = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                arrow_end = (arrow_start[0] + int(arrow_dir[0]), arrow_start[1] + int(arrow_dir[1]))
                test = pokes[i]
                # test = cv2.UMat.get(cv2.putText(cv2.UMat(test), f"Poke", (int(test.shape[1] // 3), int(5 * test .shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                #                                           float(test.shape[0] / 256), (255, 255, 255), int(test.shape[0] / 128)))
                pokes_with_arrow.append(cv2.arrowedLine(test, arrow_start, arrow_end, (255, 0, 0), max(int(src_i.shape[0] / 64),1)))
                srcs_with_arrow.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(src_i), arrow_start, arrow_end, (255, 0, 0), max(int(src_i.shape[0] / 64),1))))
                if invert_poke:
                    arrow_end_inv = (arrow_start[0] - int(arrow_dir[0]), arrow_start[1] - int(arrow_dir[1]))
                    pokes_with_arrow_inv.append(cv2.arrowedLine(test, arrow_start, arrow_end_inv, (0, 255, 0), max(int(src_i.shape[0] / 64), 1)))
                    srcs_with_arrow_inv.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(src_i), arrow_start, arrow_end, (0, 255, 0), max(int(src_i.shape[0] / 64), 1))))
            else:
                pokes_with_arrow.append(np.zeros_like(pokes[i]))
                srcs_with_arrow.append(src_i)

    poke = np.concatenate(pokes_with_arrow, axis=1)
    if invert_poke:
        poke_inv = np.concatenate(pokes_with_arrow_inv, axis=1)
        poke  = put_text_to_video_row(np.stack([*[poke] * int(math.ceil(float(seq_len)/2)),*[poke_inv]*int(seq_len/2)], axis=0),"Pokes",color=(255,255,255))
    else:
        poke = put_text_to_video_row(np.stack([poke] * seq_len, axis=0),"Poke",color=(255,255,255))


    if flows_vis is None:
        flows_vis = [poke]
    else:
        flows_vis.append(poke)


    srcs = np.concatenate(srcs_with_arrow,axis=1)
    srcs = cv2.UMat.get(cv2.putText(cv2.UMat(srcs), f"Sequence length {seq_len}", (int(srcs.shape[1] // 3), int(srcs.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                   float(srcs.shape[0] / 256), (255, 0, 0), int(srcs.shape[0] / 128)))
    if invert_poke:
        srcs_inv = np.concatenate(srcs_with_arrow_inv, axis=1)
        srcs_inv = cv2.UMat.get(cv2.putText(cv2.UMat(srcs_inv), f"Sequence length {seq_len}", (int(srcs_inv.shape[1] // 3), int(srcs_inv.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                        float(srcs_inv.shape[0] / 256), (255, 0, 0), int(srcs_inv.shape[0] / 128)))
        srcs = np.stack([*[srcs] * int(math.ceil(float(seq_len)/2)),*[srcs_inv]*int(seq_len/2)],axis=0)
    else:
        srcs = np.stack([srcs]*seq_len,axis=0)
    srcs = put_text_to_video_row(srcs,"Input Image",display_frame_nr=display_frame_nr)


    pred = ((pred.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    pred = np.concatenate(list(pred),axis=2)
    pred = put_text_to_video_row(pred, "Predicted Video",display_frame_nr=display_frame_nr)


    tgt = ((tgt.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    tgt = np.concatenate(list(tgt),axis=2)
    tgt = put_text_to_video_row(tgt,"Groundtruth Video",display_frame_nr=display_frame_nr)


    full = np.concatenate([srcs,pred,tgt,*flows_vis],axis=1)
    if logwandb:
        full = np.moveaxis(full,[0,1,2,3],[0,2,3,1])

    return full

def put_text_to_video_row(video_row,text, color = None,display_frame_nr=False):

    written = []
    for i,frame in enumerate(video_row):
        current = cv2.UMat.get(cv2.putText(cv2.UMat(frame), text, (int(frame.shape[1] // 3), frame.shape[0] - int(frame.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(frame.shape[0] / 256), (255, 0, 0) if color is None else color, int(frame.shape[0] / 128)))
        if display_frame_nr:
            current = cv2.UMat.get(cv2.putText(cv2.UMat(current), str(i+1), (int(frame.shape[1] / 32), frame.shape[0] - int(frame.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                           float(frame.shape[0] / 256), (255, 0, 0) if color is None else color, int(frame.shape[0] / 128)))

        written.append(current)
    return np.stack(written)


def make_animated_grid(src, poke, pred, tgt, n_logged, flow=None, length_divisor=5,logwandb=True):
    # visualize flows
    pokes = vis_flow(poke[:n_logged])
    pokes_with_arrow = []
    for i,poke_p in enumerate(poke[:n_logged]):
        poke_points = np.nonzero(pokes[i].any(-1) > 0)
        min_y = np.amin(poke_points[0])
        max_y = np.amax(poke_points[0])
        min_x = np.amin(poke_points[1])
        max_x = np.amax(poke_points[1])
        # plot mean direction of flow in poke region
        avg_flow = np.mean(poke_p[:,min_y:max_y,min_x:max_x].cpu().numpy(), axis=(1, 2))
        arrow_dir = avg_flow / np.linalg.norm(avg_flow) * (poke_p.shape[1] / length_divisor)
        arrow_start = (int((min_x+max_x)/2),int((min_y+max_y)/2))
        arrow_end = (arrow_start[0]+int(arrow_dir[0]),arrow_start[1]+int(arrow_dir[1]))
        test = pokes[i]
        pokes_with_arrow.append(cv2.arrowedLine(test,arrow_start,arrow_end,(0,0,255),2))



    poke = np.concatenate(pokes_with_arrow, axis=1)
    flows_vis = [np.stack([poke]*3,axis=0)]

    if flow is not None:
        flows = vis_flow(flow[:n_logged])
        flows_with_rect = []
        for i,(poke_p,flow) in enumerate(zip(pokes,flows)):
            poke_points = np.nonzero(poke_p.any(-1) > 0)
            min_y = np.amin(poke_points[0])
            max_y = np.amax(poke_points[0])
            min_x = np.amin(poke_points[1])
            max_x = np.amax(poke_points[1])
            # draw rect
            flow_with_rect = cv2.rectangle(flow,(min_x,min_y),(max_x,max_y),(255,255,255),max(1,int(flow.shape[0]//64)))


            flows_with_rect.append(flow_with_rect)

        flow_cat = np.concatenate(flows_with_rect,axis=1)
        flows_vis.insert(0,np.stack([flow_cat]*3,axis=0))




    # visualize images
    src = ((src.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    pred = ((pred.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    tgt = ((tgt.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    src = np.concatenate(list(src),axis=1)
    pred = np.concatenate(list(pred), axis=1)
    tgt = np.concatenate(list(tgt), axis=1)
    src = cv2.UMat.get(cv2.putText(cv2.UMat(src), "Source", (int(src.shape[0] // 4), 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   float(src.shape[0] / 256), (0, 0, 0), int(src.shape[0] / 128)))
    pred = cv2.UMat.get(cv2.putText(cv2.UMat(pred), "Predicted", (int(pred.shape[0] // 4), 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    float(pred.shape[0] / 256), (0, 0, 0), int(pred.shape[0] / 128)))
    tgt = cv2.UMat.get(cv2.putText(cv2.UMat(tgt), "Target", (int(tgt.shape[0] // 4), 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   float(tgt.shape[0] / 256), (0, 0, 0), int(tgt.shape[0] / 128)))

    animation = np.stack([src,pred,tgt],axis=0)
    # this generates a video grid which can be used by wandb.Video()
    full = np.concatenate([animation,*flows_vis],axis=1)
    # wandb requires video to have shape (time, channels, height, width)
    if logwandb:
        full = np.moveaxis(full,[0,1,2,3],[0,2,3,1])

    return full

def make_generic_grid(data, dtype, n_logged):
    from utils.visualizer import FlowVisualizer
    visualizer = FlowVisualizer()
    final_data = []
    assert(len(data)==len(dtype))
    for i, batch in enumerate(data):
        if dtype[i] == "flow":
            add = batch.permute(0, 2, 3, 1).cpu().numpy()[:n_logged]
            add -= add.min()
            add /= add.max()
            add = (add * 255.0).astype(np.uint8)
            image = np.concatenate(
                [add, np.expand_dims(np.zeros_like(add).sum(-1), axis=-1)], axis=-1).astype(np.uint8)
        elif dtype[i] == "flow_3D":
            add = batch.permute(0, 2, 3, 1).cpu().numpy()[:n_logged]
            add -= add.min()
            add /= add.max()
            add = (add * 255.0).astype(np.uint8)
            image = add
        elif dtype[i] == "img":
            image = ((batch.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
        elif dtype[i] == "diff_flow_amplitude":
            generated = batch[0][:n_logged].detach().cpu()
            target = batch[1][:n_logged].detach().cpu()
            image = visualizer.create_diff_amplitude(
                visualizer.make_3d_to_2d(generated), visualizer.make_3d_to_2d(target))
            image = (image*255).astype(np.uint8)[:, None].repeat(3, axis=1).transpose(0, 2, 3, 1)
        elif dtype[i] == "diff_flow_direction":
            generated = batch[0][:n_logged].detach().cpu()
            target = batch[1][:n_logged].detach().cpu()
            image = visualizer.create_diff_direction(
                visualizer.make_3d_to_2d(generated), visualizer.make_3d_to_2d(target))
            image = (image * 255).astype(np.uint8)[:, None].repeat(3, axis=1).transpose(0, 2, 3, 1)
        elif dtype[i] == "diff_flow_clipped":
            generated = batch[0][:n_logged].permute(0, 2, 3, 1).cpu().numpy()
            target = batch[1][:n_logged].permute(0, 2, 3, 1).cpu().numpy()
            image = np.sum(np.abs(generated-target), axis=-1)
            image = (image[:, :, :, None]).astype(np.uint8)
            image = np.clip(image, 0, 255)
            image = np.repeat(image, 3, axis=-1)
        elif dtype[i] == "diff_scaled":
            generated = batch[0][:n_logged].permute(0, 2, 3, 1).cpu().numpy()
            target = batch[1][:n_logged].permute(0, 2, 3, 1).cpu().numpy()
            image = np.sum(np.abs(generated-target), axis=-1)
            image /= image.max(axis=0)
            image = (image[:, :, :, None]*255.0).astype(np.uint8)
            image = np.repeat(image, 3, axis=-1)
        image = np.concatenate([s for s in image], axis=1)
        final_data.append(image)
    grid = np.concatenate(final_data, axis=0)
    return grid

def make_img_grid(appearance, shape, pred, tgt= None, n_logged=4, target_label="Target Images",
                  label_app = "Appearance Images", label_gen = "Generated Images", label_shape = "Shape Images"):
    """
    :param appearance:
    :param shape:
    :param pred:
    :param tgt:
    :param n_logged:
    :return:
    """
    appearance = ((appearance.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    shape = ((shape.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    pred = ((pred.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(
        np.uint8)[:n_logged]
    if tgt is not None:
        tgt = ((tgt.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(
            np.uint8)[:n_logged]
        tgt = np.concatenate([t for t in tgt], axis=1)
        tgt = cv2.UMat.get(cv2.putText(cv2.UMat(tgt), target_label , (int(tgt.shape[1] // 3), tgt.shape[0] - int(tgt.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(tgt.shape[0] / 256), (255, 0, 0), int(tgt.shape[0] / 128)))

    appearance = np.concatenate([s for s in appearance], axis=1)
    appearance = cv2.UMat.get(cv2.putText(cv2.UMat(appearance),label_app, (int(appearance.shape[1] // 3), appearance.shape[0] - int(appearance.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(appearance.shape[0] / 256), (255, 0, 0), int(appearance.shape[0] / 128)))
    shape = np.concatenate([f for f in shape], axis=1)
    shape = cv2.UMat.get(cv2.putText(cv2.UMat(shape), label_shape, (int(shape.shape[1] // 3), shape.shape[0] - int(shape.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(shape.shape[0] / 256), (255, 0, 0), int(shape.shape[0] / 128)))
    pred = np.concatenate([p for p in pred], axis=1)
    pred = cv2.UMat.get(cv2.putText(cv2.UMat(pred), label_gen, (int(pred.shape[1] // 3), pred.shape[0] - int(pred.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(pred.shape[0] / 256), (255, 0, 0), int(pred.shape[0] / 128)))


    if tgt is None:
        grid = np.concatenate([appearance, shape, pred], axis=0)
    else:
        grid = np.concatenate([appearance, shape, pred, tgt], axis=0)
    return grid


def scale_img(img):
    """
    Rescales an image to the actual pixel domain in [0,255] and converts it to the dtype
    :param img:
    :return:
    """
    img = (img + 1.) * 127.5
    if isinstance(img, torch.Tensor):
        img = img.to(torch.uint8)
    else:
        # assumed to be numpy array
        img = img.astype(np.uint8)
    return img

def human_graph_cut_map(img, poke_size):
    import cv2
    import matplotlib.pyplot as plt

    # make backgound foreground segementation
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (int(img.shape[1] / 5),
            poke_size,
            int(3. * img.shape[1] / 5),
            int(img.shape[0] - 2 * poke_size))

    fgm = np.zeros((1, 65), dtype=np.float64)
    bgm = np.zeros((1, 65), dtype=np.float64)

    mask2, fgm, bgm = cv2.grabCut(img, mask, rect, fgm, bgm, 5, cv2.GC_INIT_WITH_RECT)
    mask3 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype(np.bool)
    tuples = np.where(mask3[:, :])
    return tuples

    new_img = np.zeros_like(img)
    for t in tuples:
        for i in range(3):
            new_img[t[0], t[1], i] = 255
    # show the output frame
    plt.imshow(new_img)
    plt.show()


# human_NN_map_weights = "/export/home/jsieber/poking/models/mask-rcnn-coco/frozen_inference_graph.pb"
# human_NN_map_classes = "/export/home/jsieber/poking/models/mask-rcnn-coco/object_detection_classes_coco.txt"
# human_NN_map_config = "/export/home/jsieber/poking/models/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
# human_NN_map_LABELS = open(human_NN_map_classes).read().strip().split("\n")
#
# human_NN_map_net = cv2.dnn.readNetFromTensorflow(human_NN_map_weights, human_NN_map_config)
# def human_NN_map(frame, conf=0.5, threshold=0.3):
#
#
#     # modified from https://www.pyimagesearch.com/2018/11/26/instance-segmentation-with-opencv/
#     before_width = frame.shape[1]
#     frame = imutils.resize(frame, width=600)
#     (H, W) = frame.shape[:2]
#
#     # construct a blob from the input image and then perform a
#     # forward pass of the Mask R-CNN, giving us (1) the bounding
#     # box coordinates of the objects in the image along with (2)
#     # the pixel-wise segmentation for each specific object
#     blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
#     human_NN_map_net.setInput(blob)
#     (boxes, masks) = human_NN_map_net.forward(["detection_out_final",
#                                   "detection_masks"])
#
#     # sort the indexes of the bounding boxes in by their corresponding
#     # prediction probability (in descending order)
#     idxs = np.argsort(boxes[0, 0, :, 2])[::-1]
#
#     # initialize the mask, ROI, and coordinates of the person for the
#     # current frame
#     mask = None
#     roi = None
#     coords = None
#
#     # loop over the indexes
#     for i in idxs:
#         # extract the class ID of the detection along with the
#         # confidence (i.e., probability) associated with the
#         # prediction
#         classID = int(boxes[0, 0, i, 1])
#         confidence = boxes[0, 0, i, 2]
#
#         # if the detection is not the 'person' class, ignore it
#         if human_NN_map_LABELS[classID] != "person":
#             continue
#         # filter out weak predictions by ensuring the detected
#         # probability is greater than the minimum probability
#         if confidence > conf:
#             # scale the bounding box coordinates back relative to the
#             # size of the image and then compute the width and the
#             # height of the bounding box
#             box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
#             (startX, startY, endX, endY) = box.astype("int")
#             coords = (startX, startY, endX, endY)
#             boxW = endX - startX
#             boxH = endY - startY
#
#             # extract the pixel-wise segmentation for the object,
#             # resize the mask such that it's the same dimensions of
#             # the bounding box, and then finally threshold to create
#             # a *binary* mask
#             mask = masks[i, classID]
#             mask = cv2.resize(mask, (boxW, boxH),
#                               interpolation=cv2.INTER_NEAREST)
#             mask = (mask > threshold)
#
#             # extract the ROI and break from the loop (since we make
#             # the assumption there is only *one* person in the frame
#             # who is also the person with the highest prediction
#             # confidence)
#             roi = frame[startY:endY, startX:endX][mask]
#             break
#
#     # initialize our output frame
#     output = frame.copy()
#
#     # if the mask is not None *and* we are in privacy mode, then we
#     # know we can apply the mask and ROI to the output image
#     if mask is not None:
#         # blur the output frame
#         output = np.zeros_like(output)
#
#         # add the ROI to the output frame for only the masked region
#         (startX, startY, endX, endY) = coords
#         roi = np.ones_like(roi)*255
#         output[startY:endY, startX:endX][mask] = roi
#     output = imutils.resize(output, width=before_width)
#
#     tuples = np.where(output[:, :, 0] > 0)
#     return tuples
#
#     new_img = np.zeros_like(output)
#     for t in tuples:
#         for i in range(3):
#             new_img[t[0], t[1], i] = 255
#     # show the output frame
#     plt.imshow(new_img)
#     plt.show()


def make_hist(hist, title, ylabel, xlabel="Frame number", bins_edges = None):
    plt.ioff()
    if bins_edges is None:
        bins_edges = np.arange(0, len(hist) + 1).astype(np.float)
    else:
        assert len(list(bins_edges)) == len(list(hist)) + 1
    centroids = (bins_edges[1:] + bins_edges[:-1]) / 2

    hist_, bins_, _ = plt.hist(
        centroids,
        bins=len(hist),
        weights=np.asarray(hist),
        range=(min(bins_edges), max(bins_edges)),
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    wandb.log({title: wandb.Image(plt)})
    plt.close()


def make_plot(x,y,title,ylabel,xlabel="frame idx", savename=None):
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(x,y,'rv-')
    ax.set(xlabel=xlabel, ylabel=ylabel, title = title)
    ax.grid()

    if savename is None:
        wandb.log({title:wandb.Image(plt)})
    else:
        fig.savefig(savename)

    plt.close()


if __name__=="__main__":

    frame_path = "/export/data/ablattma/Datasets/iPER/processed/001_10_1/frame_277.png"
    frame = cv2.imread(frame_path)
    frame = imutils.resize(frame, width=128)

    import time
    for i in range(3):
        start_time = time.time()
        human_graph_cut_map(frame, 15)
        print("--- %s seconds ---" % (time.time() - start_time))

    for i in range(3):
        start_time = time.time()
        human_NN_map(frame)
        print("--- %s secondss ---" % (time.time() - start_time))
