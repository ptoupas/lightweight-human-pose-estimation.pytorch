import argparse
import math
import time

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from trt_engine import ONNXClassifierWrapper, TrtModel


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img,
                                    pad[0],
                                    pad[2],
                                    pad[1],
                                    pad[3],
                                    cv2.BORDER_CONSTANT,
                                    value=pad_value)
    return padded_img, pad


class ImageReader(object):

    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(
                self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):

    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net,
               img,
               net_input_height_size,
               stride,
               upsample_ratio,
               cpu,
               pad_value=(0, 0, 0),
               img_mean=np.array([128, 128, 128], np.float32),
               img_scale=np.float32(1 / 256),
               trt=False):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0),
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [
        net_input_height_size,
        max(scaled_img.shape[1], net_input_height_size)
    ]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0,
                                                      1).unsqueeze(0).float()
    if not trt:
        if not cpu:
            tensor_img = tensor_img.cuda()
        stages_output = net(tensor_img)
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(),
                                (1, 2, 0))
    else:
        trt_img = tensor_img.detach().numpy().astype(np.float32)
        stages_output = net(trt_img)
        stage2_heatmaps = stages_output[-2].reshape(1, 19, 32, 43)
        heatmaps = np.transpose(stage2_heatmaps.squeeze(), (1, 2, 0))

    heatmaps = cv2.resize(heatmaps, (0, 0),
                          fx=upsample_ratio,
                          fy=upsample_ratio,
                          interpolation=cv2.INTER_CUBIC)

    if not trt:
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(),
                            (1, 2, 0))
    else:
        stage2_pafs = stages_output[-1].reshape(1, 38, 32, 43)
        pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0),
                      fx=upsample_ratio,
                      fy=upsample_ratio,
                      interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth, trt=False):
    if not trt:
        net = net.eval()
        if not cpu:
            net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    prev_frame_time = 0
    new_frame_time = 0
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net,
                                                img,
                                                height_size,
                                                stride,
                                                upsample_ratio,
                                                cpu,
                                                trt=trt)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx],
                                                     all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type,
                                                      pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride /
                                        upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride /
                                        upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(
                img, (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id),
                            (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        cv2.putText(img, f"FPS={fps:.2f}", (7, 70), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path',
                        type=str,
                        required=True,
                        help='path to the checkpoint')
    parser.add_argument('--height-size',
                        type=int,
                        default=256,
                        help='network input layer height size')
    parser.add_argument('--video',
                        type=str,
                        default='',
                        help='path to video file or camera id')
    parser.add_argument('--images',
                        nargs='+',
                        default='',
                        help='path to input image(s)')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='run network inference on cpu')
    parser.add_argument('--track',
                        type=int,
                        default=0,
                        help='track pose id in video')
    parser.add_argument('--smooth',
                        type=int,
                        default=0,
                        help='smooth pose keypoints')
    parser.add_argument('--export-onnx',
                        action='store_true',
                        help='export onnx model')
    parser.add_argument('--trt-inferece',
                        action='store_true',
                        help='do inference using TensorRT')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    if not args.trt_inferece:
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)

        if args.export_onnx:
            net.cpu().eval()

            input_tensor = torch.randn((1, 3, 256, 344))

            torch.onnx.export(
                net,
                input_tensor,
                "checkpoints/pose_estimation_with_mobilenet.onnx",
                export_params=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                opset_version=11,
                do_constant_folding=False,
                training=torch.onnx.TrainingMode.PRESERVE)
            exit()

        run_demo(net, frame_provider, args.height_size, args.cpu, args.track,
                 args.smooth)
    else:
        trt_model = TrtModel(
            "checkpoints/pose_estimation_with_mobilenet_fp32.engine",
            max_batch_size=1,
            dtype=np.float32)
        run_demo(trt_model,
                 frame_provider,
                 args.height_size,
                 args.cpu,
                 args.track,
                 args.smooth,
                 trt=True)
