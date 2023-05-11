import logging
import copy
import time
from threading import Thread
from PIL import Image
import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from imutils.video import FPS

from postprocess import yolov5_post_process
from config import CLASSES

logger = logging.getLogger()

def get_host(device_compatible_node):
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(device_compatible_node) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK356x'
        except IOError:
            print('Read device node {} failed.'.format(device_compatible_node))
            exit(-1)
    else:
        host = os_machine
    return host

def draw(image, boxes, scores, classes,class_dict):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
#        print('class: {}, score: {}'.format(CLASSES[cl], score))
#        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(class_dict[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class Camera:
    def __init__(self, output,config,yolo5=False,time_sec=60):
        self.output = output
        self.width = config.width
        self.height = config.height
        self.framerate = config.framerate
        self.yolo5 = yolo5
        self.config = config

        self.fps = FPS().start()
        self.fps_time_sec = time.time()
        self.elapsed_time_sec = time.time()
        self.time_sec = time_sec
        self.prev_frame_time = 0
        self.new_frame_time = 0
        if yolo5:
            host_name = get_host(config.device_compatible_node)
            if host_name == 'RK356x':
                rknn_model = config.rk356x_rknn_model
            elif host_name == 'RK3588':
                rknn_model = config.rk3588_rknn_model
            else:
                logger.info("This demo cannot run on the current platform: {}".format(host_name))
                exit(-1)
            
            self.rknn_lite = RKNNLite()
            # load RKNN model
            logger.info('--> Load RKNN model')
            ret = self.rknn_lite.load_rknn(rknn_model)
            if ret != 0:
                logger.info('Load RKNN model failed')
                exit(ret)
            logger.info('done')


            # init runtime environment
            logger.info('--> Init runtime environment')
            # run on RK356x/RK3588 with Debian OS, do not need specify target.
            if host_name == 'RK3588':
                ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            else:
                ret = self.rknn_lite.init_runtime()
            if ret != 0:
                logger.info('Init runtime environment failed')
                exit(ret)
            logger.info('done')
            
    def __enter__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
        self.stop_capture = False
        self.thread = Thread(target=self.capture)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_capture = True
        self.thread.join()
        self.cap.release()
        if self.yolo5:
            self.rknn_lite.release()

            # stop the timer and display FPS information
            self.fps.stop()
            logger.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
            logger.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
    def capture(self):
        frame_duration = 1. / self.framerate
        while not self.stop_capture:
            #Show FPS in Pic
            self.new_frame_time = time.time()
            show_fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            show_fps = int(show_fps)
            show_fps = str("FPS:{}".format(show_fps))
            elapsed_start = time.time()
            
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_out = copy.deepcopy(frame)
                if not self.yolo5:
                    pass
                else:
                    frame_out, ratio, (dw, dh) = letterbox(frame_out, new_shape=(self.config.img_size,self.config.img_size))

                    # Inference
                    outputs = self.rknn_lite.inference(inputs=[frame_out])

                    # post process
                    input0_data = outputs[0]
                    input1_data = outputs[1]
                    input2_data = outputs[2]

                    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
                    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
                    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

                    input_data = list()
                    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
                    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
                    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

                    #Disable Enable YOLO Post process
                    boxes, classes, scores = yolov5_post_process(input_data,self.config.img_size,self.config.obj_thresh,self.config.nms_thresh)
                    
                    img_1 = frame
                    if boxes is not None:

                        # post process
                        input0_data = outputs[0]
                        input1_data = outputs[1]
                        input2_data = outputs[2]

                        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
                        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
                        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

                        input_data = list()
                        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
                        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
                        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))


                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                        draw(frame_out, boxes, scores, classes,CLASSES)
                        
                # show FPS in Frame
                cv2.putText(frame_out, show_fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

                old_h, old_w = frame.shape[:2]
                ratio_h, ratio_w = ratio
                frame_out = cv2.resize(frame_out, (int(old_h/ratio_h), int(old_w/ratio_w)), interpolation=cv2.INTER_LINEAR)
                fps_time_now = time.time()

                if fps_time_now-self.fps_time_sec>self.time_sec:
                    self.time_sec_counter = fps_time_now
                    logger.info("FPS:%s"%str(show_fps))
                img = Image.fromarray(frame_out)
                img.save(self.output, format='JPEG')
                # update the FPS counter
                self.fps.update()
            elapsed = time.time() - elapsed_start
            elapsed_time_now = time.time()
            if elapsed_time_now-self.elapsed_time_sec>self.time_sec:
                logging.debug("Average frame acquisition time: %.2f" % elapsed)
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)


