
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from strategy import getgrasp
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
import warnings
warnings.filterwarnings('ignore')
import glob
import cv2
import pyzed.sl as sl    #ZED2i camera
import numpy as np
import math
import time
import socket
import json
import ctypes
import tactile_recognition
import matlab
from ctypes import *
from decodedata import decodedata
from sercommunication import Sercommunication
import threading
import datetime
import random
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

#gesture definition
eMotionType_HOME = 1
eMotionType_READY = 2
eMotionType_GRAVITY_COMP = 3
jointnum = 16
start_pos=(c_double * jointnum)( -0.10261019060953777,0.7316745685072835,0.029735652123006186,-0.11201908352010093,
                                    -0.0029291836419677738,0.7462317235764568,-0.10047987523356122,0.10624947937683107,
                                    0.0663948292179362,0.7600787735203044,-0.054145515806070965,-0.06186790904398601,
                                    1.29523, 0.27543, -0.24374,-0.02112)
grap_9 = (c_double * jointnum)(-0.11512579344340008,1.5751906942533367,0.1794790704260254,-0.19181714697855634,
                                0.0080774458005778,1.5211339415879317,0.3511469844734701,-0.2189786680222575,
                                0.114859504021403,1.4812792914290367,0.3602008248213705,-0.19874067195048017,
                                1.5091509175980635,0.0016376791522217,0.5323935515327963,0.002396604797973633)
horizontheta = math.pi/12
handlength = 50*math.cos(horizontheta)
set_Gains_kd = (c_double * 16)(50, 50, 55, 40, 50, 50, 55, 40, 50, 50, 55, 40, 100, 100, 50, 40)
kp_unit = np.array([100, 160, 180, 100, 100, 160, 180, 100, 100, 160, 180, 100, 200, 140, 180, 160])
toolHcam = np.array([[0.9997224225736, -0.0150586444581, 0.0181194655197, -67.9928346753286],
                         [0.0149080055133, 0.9998534149768, 0.0084202096693, -124.0658352692683],
                         [-0.0182436064211, -0.0081477473173, 0.9998003725937, -35.3666332794547],
                         [0, 0, 0, 1]])        #The pose of camera trlative to the end of robotic arm
global initfinish
global threadrun
global touchstate
global handstate
initfinish = False
touchstate = 0   #state of sensors，0:no touch，1:touch，2:slip
handstate = 0    #state of hand，0:open，1:grasp，2:move
read_byte_num = 146  #bytes read

#settings of robotic arm
def connectETController(ip, port=8055):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip, port))
        return (True, sock)
    except Exception as e:
        sock.close()
        return (False, None)
def disconnectETController(sock):
    if sock:
        sock.close()
        sock = None
    else:
        sock = None
def sendCMD(sock, cmd, params=None, id=1):
    if (not params):
        params = []
    else:
        params = json.dumps(params)
    sendStr = "{{\"method\":\"{0}\",\"params\":{1},\"jsonrpc\":\"2.0\",\"id\":{2}}}".format(cmd, params, id) + "\n"
    try:
        # print(sendStr)
        sock.sendall(bytes(sendStr, "utf-8"))
        ret = sock.recv(1024)
        jdata = json.loads(str(ret, "utf-8"))
        if ("result" in jdata.keys()):
            return (True, json.loads(jdata["result"]), jdata["id"])
        elif ("error" in jdata.keys()):
            return (False, jdata["error"], jdata["id"])
        else:
            return (False, None, None)
    except Exception as e:
        return (False, None, None)
def wait_stop():
    while True:
        time.sleep(0.01)
        ret1, result1, id1 = sendCMD(sock, "getRobotState")  # getRobotstate
        if ret1:
            if result1 == 0 or result1 == 4:
                break
        else:
            print("getRobotState failed")
            break


#get image
def get_test_images(infer_dir, infer_img):
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]
    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))
    return images


#initialization of tactile sensors
def get_initial_value(com, num):
    hot_film_values_initial = [0, 0, 0, 0, 0,0,0,0]
    counter = 0
    for j in range(0, num):
        recreate = com.Read_Size(read_byte_num)
        decode_right, hotfilm_values= decodedata(recreate)
        if decode_right:
            hot_film_values_initial[0] += hotfilm_values[0]
            hot_film_values_initial[1] += hotfilm_values[1]
            hot_film_values_initial[2] += hotfilm_values[2]
            hot_film_values_initial[3] += hotfilm_values[3]
            hot_film_values_initial[4] += hotfilm_values[4]
            hot_film_values_initial[5] += hotfilm_values[5]
            hot_film_values_initial[6] += hotfilm_values[6]
            hot_film_values_initial[7] += hotfilm_values[7]
            counter += 1
        time.sleep(0.01)
    hot_film_values_initial[0] = hot_film_values_initial[0] / counter
    hot_film_values_initial[1] = hot_film_values_initial[1] / counter
    hot_film_values_initial[2] = hot_film_values_initial[2] / counter
    hot_film_values_initial[3] = hot_film_values_initial[3] / counter
    hot_film_values_initial[4] = hot_film_values_initial[4] / counter
    hot_film_values_initial[5] = hot_film_values_initial[5] / counter
    hot_film_values_initial[6] = hot_film_values_initial[6] / counter
    hot_film_values_initial[7] = hot_film_values_initial[7] / counter
    return hot_film_values_initial


#thread of tactile sensors
def run2():
    global touchstate
    global handstate
    global initfinish
    global threadrun
    global sort_flag
    initfinish = False
    threadrun = True
    sort_flag = 0
    max_len = 500
    if_full = False
    mean_num = 10
    hot_film_values_list = []

    ser = Sercommunication("com5", 115200, 0.1)
    ser.Open_Ser()
    if ser.ser.isOpen():
        print("open com success")
    else:
        print("open com failed")
    time.sleep(0.1)
    #   get base line
    hot_film_base = get_initial_value(ser, 500)
    print('Initialization complete')
    print('热膜基值：', hot_film_base)
    initfinish = True
    record_mean,mean_values = np.array(hot_film_base), np.array(hot_film_base)
    sorting_test=tactile_recognition.initialize()

    f = open(r'C:\Users\86152\PycharmProjects\visiontouchsense2\hot_film_values_list.txt', 'a+')
    f.truncate(0)
    f.close()
    while (threadrun):
        start_time = time.time()
        rec_data = ser.Read_Size(read_byte_num)
        ser.ser.flushInput()
        frame_is_right, hot_film_values = decodedata(rec_data)
        if (hot_film_values_list):
            if (hot_film_values[0] - hot_film_base[0] > 0.002 or hot_film_values[1] - hot_film_base[1] > 0.002 or hot_film_values[2] - hot_film_base[2] > 0.002 or hot_film_values[4] - hot_film_base[4] > 0.002 or hot_film_values[5] - hot_film_base[5] > 0.002 or hot_film_values[6] - hot_film_base[6] > 0.002) \
                    or (hot_film_values[3] - hot_film_base[3]  > 0.0012 or hot_film_values[7] - hot_film_base[7]  > 0.002):
                if (handstate==1):
                    touchstate = 1
                    print('grasp successfully!')
                elif(handstate==2):        #sorting based on tactile senses
                    if sort_flag==4:
                        dU = np.array(hot_film_values) - np.array(hot_film_base)
                        dU = dU.tolist()
                        xin = matlab.double(dU, size=(8, 1))
                        yout = sorting_test.tactile_recognition(xin)
                        yout = np.array(yout)
                        sort_flag = yout.argmax()+1
                    if (hot_film_values[4] - record_mean[4] > 0.00065 or hot_film_values[5] - record_mean[
                        5] > 0.00065 or hot_film_values[6] - record_mean[6] > 0.00065 or hot_film_values[7] -
                            record_mean[7] > 0.00065):
                        print('slip!', datetime.datetime.now())
                        touchstate = 2
                    else:
                        touchstate = 1
                        print('hold on!', datetime.datetime.now())
            else:
                touchstate = 0
            record_mean = mean_values
            mean_values = (mean_values * (mean_num - 1) + hot_film_values) / mean_num
        hot_film_values_list.append(hot_film_values)
        if (if_full):
            hot_film_values_list.pop(0)
        else:
            if (len(hot_film_values_list)>=max_len):
                if_full = True

        # print('hot_film_values', hot_film_values)
        f = open(r'C:\Users\86152\PycharmProjects\visiontouchsense2\hot_film_values_list.txt', 'a+')
        f.writelines(str(hot_film_values) + '\n')
        f.close()


# object detection
def check_object(cam, runtime, mat, depth, point_cloud,FLAGS,trainer):
    err = cam.grab(runtime)
    resultflag = 0
    object_list = []
    pos_list = []
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        img = mat.get_data()
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
        depth_map = depth.get_data()
        cam.retrieve_measure(point_cloud, sl.MEASURE.XYZBGRA, sl.MEM.CPU)
        point_map = point_cloud.get_data()
        (heigh, width, temp) = img.shape
        savePath = os.path.join("./demo", "V{:0>3d}.png".format(0))
        img = img[0:heigh, int((width - heigh) / 2):int((width + heigh) / 2)]
        img = cv2.resize(img, (320, 320))
        cv2.imwrite(savePath, img)
        images = get_test_images(FLAGS.infer_dir, os.path.join("./demo", "V{:0>3d}.png".format(0)))
        if FLAGS.slice_infer:
            trainer.slice_predict(
                images,
                slice_size=FLAGS.slice_size,
                overlap_ratio=FLAGS.overlap_ratio,
                combine_method=FLAGS.combine_method,
                match_threshold=FLAGS.match_threshold,
                match_metric=FLAGS.match_metric,
                draw_threshold=FLAGS.draw_threshold,
                output_dir=FLAGS.output_dir,
                save_results=FLAGS.save_results,
                visualize=FLAGS.visualize)
        else:
            Item, Box = trainer.predict(
                images,
                draw_threshold=FLAGS.draw_threshold,
                output_dir=FLAGS.output_dir,
                save_results=FLAGS.save_results,
                visualize=FLAGS.visualize)
        if (len(Box) > 0):              #Location of items
            suc, result, id = sendCMD(sock, "getRobotPose")
            [x, y, z, rx, ry, rz] = result
            Rx = np.array([[1, 0, 0],
                           [0, math.cos(rx), -math.sin(rx)],
                           [0, math.sin(rx), math.cos(rx)]])
            Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                           [0, 1, 0],
                           [-math.sin(ry), 0, math.cos(ry)]])
            Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                           [math.sin(rz), math.cos(rz), 0],
                           [0, 0, 1]])
            Rot = np.dot(np.dot(Rz, Ry), Rx)
            baseHtool = np.array([np.append(Rot[0], [x]), np.append(Rot[1], [y]), np.append(Rot[2], [z])])
            baseHtool = np.append(baseHtool, [0, 0, 0, 1])
            baseHtool = baseHtool.reshape(4, 4)
            baseHcam = np.dot(baseHtool, toolHcam)
            for i in range(len(Box)):
                print(Item[i], Box[i])
                (xmin, ymin, w, h) = Box[i]
                xmin = int(xmin * 9 / 4 + (width - heigh) / 2)
                ymin = int(ymin * 9 / 4)
                w = int(w * 9 / 4)
                h = int(h * 9 / 4)
                cX = int(xmin + w / 2)
                cY = int(ymin + h / 2)
                coincam = point_map[cY, cX]
                count = 0
                while (str(coincam[0]) == 'nan' and count < 30):
                    count += 1
                    cY += random.randint(-5, 5)
                    cX += random.randint(-5, 5)
                    if (cY < ymin):
                        cY = int(ymin + 1)
                    if cX < xmin:
                        cX = int(xmin + 1)
                    if cY > ymin + h:
                        cY = int(ymin + h)
                    if cX > xmin + w:
                        cX = int(xmin + w)
                    coincam = point_map[cY, cX]  # 由上一步给出
                if (count < 30):
                    coincam[3] = 1
                    coinbase = np.dot(baseHcam, coincam)
                    print('coinbase', coinbase)
                    object_list.append(Item[i])
                    pos_list.append(coinbase)
                    resultflag += 1
    return resultflag,object_list,pos_list

def run(FLAGS, cfg):
    thread1 = threading.Thread(target=run2)
    thread1.start()
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)
    Item=[]
    Box=[]
    global handstate
    global touchstate
    global sort_flag
    sort_flag = 0

    # initialization of car
    master = modbus_tcp.TcpMaster(host="192.168.71.50", port=502)
    master.set_timeout(0.5)
    Hold_value = master.execute(slave=17, function_code=cst.WRITE_SINGLE_COIL, starting_address=15,
                                output_value=[1])

    # initialization of camera
    print("Camera Running...")
    init = sl.InitParameters(depth_minimum_distance=0.0)
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 5)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 58)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 40)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)

    # initialization of robotic hand
    testdll = cdll.LoadLibrary(r'C:\Users\86152\PycharmProjects\testrobot\Peak Release\myAllegroHand.dll')
    pBHand = testdll.PrintInstruction()
    testdll.InitVarAndMemory()
    if (testdll.CreateBHandAlgorithm() and testdll.OpenCAN()):
        print('创造手对象并打开CAN')
        testdll.HandMotionUsePredifinedType(eMotionType_GRAVITY_COMP)

    # initialization of robotic arm
    if (conSuc):
        suc, result, id = sendCMD(sock, "getRobotState")
        if (result == 4):
            suc, result, id = sendCMD(sock, "clearAlarm", {"force": 0})
            time.sleep(0.5)
        suc, result, id = sendCMD(sock, "getMotorStatus")
        if (result == 0):
            suc, result, id = sendCMD(sock, "syncMotorStatus")
            time.sleep(0.5)
        suc, result, id = sendCMD(sock, "getServoStatus")
        if (result == 0):
            suc, result, id = sendCMD(sock, "set_servo_status", {"status": 1})
            time.sleep(1)
        suc, result, id = sendCMD(sock, "get_transparent_transmission_state")
        if (result == 1):
            suc, result, id = sendCMD(sock, "tt_clear_servo_joint_buf", {"clear": 0})
        wait_stop()

    # robot starts to observe and move
    global initfinish
    while (initfinish == False):
        time.sleep(0.1)
    jointlist = [55]
    for joint1 in jointlist:
        point = [joint1, -95, 88, -140, 94, -2, 0, 0]
        if (conSuc):
            suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 30})
        wait_stop()
        resultflag, object_list, pos_list = check_object(cam, runtime, mat, depth, point_cloud, FLAGS, trainer)
        if (resultflag!=0):
            print(object_list)
            print(pos_list)
            break
    if resultflag!=0:
        near_pos = []
        near_dis = 4000
        for i in range(resultflag):
            pos = pos_list[i]
            if (pos[0]*pos[0]+pos[1]*pos[1]<near_dis*near_dis):
                near_pos = pos
                near_dis = math.sqrt(pos[0]*pos[0]+pos[1]*pos[1])
    print('near_pos:',near_pos)
    time.sleep(1)
    master = modbus_tcp.TcpMaster(host="192.168.71.50", port=502)
    master.set_timeout(0.5)
    Hold_value = master.execute(slave=17, function_code=cst.WRITE_SINGLE_COIL, starting_address=15,
                                output_value=[1])
    time.sleep(0.1)
    forward_unit = near_pos[1] - 700
    print(forward_unit)
    point = [45, -95, 88, -105, 94, -2, 0, 0]
    if (conSuc):
        suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 25})
    for i in range(round(forward_unit/100 * 10)):
        Hold_value = master.execute(slave=17, function_code=cst.WRITE_MULTIPLE_REGISTERS, starting_address=40022,
                                    output_value=[-100, 0, 0])
        time.sleep(0.1)
    wait_stop()
    resultflag, object_list, pos_list = check_object(cam, runtime, mat, depth, point_cloud, FLAGS, trainer)
    near_pos = []
    if (resultflag != 0):
        print(object_list)
        print(pos_list)
        near_dis = 4000
        for i in range(resultflag):
            pos = pos_list[i]
            if (pos[0] * pos[0] + pos[1] * pos[1] < near_dis * near_dis):
                near_pos = pos
                near_dis = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    if (near_pos != []):
        print('near_pos:', near_pos)  #
        forward_dis = near_pos[1]+100
        print(forward_dis)
        point = [-37, -95, 88, -89, 94, -34, 0, 0]
        if (conSuc):
            suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 30})
        for i in range(round(forward_dis/100 * 10)):  # 以100mm为单位
            Hold_value = master.execute(slave=17, function_code=cst.WRITE_MULTIPLE_REGISTERS, starting_address=40022,
                                        output_value=[-100, 0, 0])
            time.sleep(0.1)
        wait_stop()
        if (testdll.CreateBHandAlgorithm() and testdll.OpenCAN()):
            testdll.HandMotion(start_pos)
            kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16 = tuple(kp_unit * 1.8)
            kp_now = (c_double * 16)(1000, kp2, kp3, kp4, 1000, kp6, kp7, kp8, 1000, kp10, kp11, kp12, 2000, 1400, kp15,
                                     kp16)
            testdll.setGains(kp_now, set_Gains_kd)
            time.sleep(0.1)
        wait_stop()
        while resultflag > 0:      #robot start to clean the desk
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                img = mat.get_data()
                cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
                depth_map = depth.get_data()
                cam.retrieve_measure(point_cloud, sl.MEASURE.XYZBGRA, sl.MEM.CPU)
                point_map = point_cloud.get_data()
                savePath = os.path.join("./demo", "V{:0>3d}.png".format(1))
                (heigh, width, temp) = img.shape
                img = img[0:heigh, int((width - heigh) / 2):int((width + heigh) / 2)]
                img = cv2.resize(img, (320, 320))
                cv2.imwrite(savePath, img)
                images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
                if FLAGS.slice_infer:
                    trainer.slice_predict(
                        images,
                        slice_size=FLAGS.slice_size,
                        overlap_ratio=FLAGS.overlap_ratio,
                        combine_method=FLAGS.combine_method,
                        match_threshold=FLAGS.match_threshold,
                        match_metric=FLAGS.match_metric,
                        draw_threshold=FLAGS.draw_threshold,
                        output_dir=FLAGS.output_dir,
                        save_results=FLAGS.save_results,
                        visualize=FLAGS.visualize)
                else:
                    Item, Box = trainer.predict(
                        images,
                        draw_threshold=FLAGS.draw_threshold,
                        output_dir=FLAGS.output_dir,
                        save_results=FLAGS.save_results,
                        visualize=FLAGS.visualize)
                resultflag = len(Box)
                print('resultflag:', resultflag)
                if (resultflag > 0):   #location and get grasping strategy
                    suc, result, id = sendCMD(sock, "getRobotPose")
                    [x, y, z, rx, ry, rz] = result
                    Rx = np.array([[1, 0, 0],
                                   [0, math.cos(rx), -math.sin(rx)],
                                   [0, math.sin(rx), math.cos(rx)]])
                    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                                   [0, 1, 0],
                                   [-math.sin(ry), 0, math.cos(ry)]])
                    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                                   [math.sin(rz), math.cos(rz), 0],
                                   [0, 0, 1]])
                    Rot = np.dot(np.dot(Rz, Ry), Rx)
                    baseHtool = np.array([np.append(Rot[0], [x]), np.append(Rot[1], [y]), np.append(Rot[2], [z])])
                    baseHtool = np.append(baseHtool, [0, 0, 0, 1])
                    baseHtool = baseHtool.reshape(4, 4)
                    print("baseHtool:", baseHtool)
                    baseHcam = np.dot(baseHtool, toolHcam)
                    for i in range(len(Box)):
                        print(Item[i], Box[i])
                        (xmin, ymin, w, h) = Box[i]
                        xmin = int(xmin * 9 / 4 + (width - heigh) / 2)
                        ymin = int(ymin * 9 / 4)
                        w = int(w * 9 / 4)
                        h = int(h * 9 / 4)
                        (flag, cX, cY, rotZ) = getgrasp(Item[i], depth_map, point_map, baseHcam, xmin, ymin, w, h)  # get strategy
                        if (flag != 0):
                            cX = int(cX)
                            cY = int(cY)
                            coincam = point_map[cY, cX]
                            coincam[3] = 1
                            print('coincam:', coincam)
                            coinbase = np.dot(baseHcam, coincam)
                            print('coinbase', coinbase)
                            point = coinbase
                            if (Item[i] == 'cup'):
                                containwater = False
                            point[2] = point[2] + 300
                            tempx = point[0]
                            tempy = point[1]
                            point = np.delete(point, [3])
                            print('flag:', flag)
                            if (flag == 1):
                                point[1] = tempy - handlength * math.cos(rotZ) * 1.2
                                point[0] = tempx + handlength * math.sin(rotZ) * 1.2
                                point = np.append(point, [-math.pi / 2 - horizontheta, 0, rotZ])
                            elif (flag == 2):
                                point[0] = tempx + handlength * 0.35
                                point[1] = tempy - handlength * 2.7
                                point = np.append(point,
                                                  [-math.pi / 180 * 96, math.pi / 180 * 57.6, -math.pi / 180 * 52.5])
                            else:
                                point[0] = tempx + handlength * 0.1
                                point[1] = tempy - handlength * 3.04
                                point = np.append(point,
                                                  [-math.pi / 180 * 127.5, math.pi / 180 * 81.7, -math.pi / 180 * 89])
                            point = point.tolist()
                            print("point:", point)
                            if (conSuc):
                                suc, result_joint, id = sendCMD(sock, "inverseKinematic", {"targetPose": point})
                                print('result_joint', result_joint)
                                time.sleep(0.01)
                                suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": result_joint, "speed": 40})
                                wait_stop()
                                if (flag == 2):
                                    point[2] = point[2] - 300
                                elif (flag == 1):
                                    point[2] = point[2] - 180
                                else:
                                    point[2] = point[2] - 330
                                print('height of Z:', point[2])
                                suc, result_joint, id = sendCMD(sock, "inverseKinematic", {"targetPose": point})
                                suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": result_joint, "speed": 20})
                                print('result_joint', result_joint)
                                wait_stop()
                                if (testdll.CreateBHandAlgorithm() and testdll.OpenCAN()):  # grap
                                    time.sleep(0.1)
                                    if (Item[i] == 'paper' or Item[i] == 'plastic bag'):
                                        force_unit = 3
                                    elif Item[i] == 'cup':
                                        force_unit = 2.4
                                    else:
                                        force_unit = 1.6
                                    testdll.HandMotion(grap_9)
                                    kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16 = tuple(
                                        kp_unit * force_unit)
                                    kp_now = (c_double * 16)(2000, kp2, kp3, kp4, 2000, kp6, kp7, kp8, 2000, kp10, kp11,
                                                             kp12, 4000, 2800, kp15, kp16)
                                    testdll.setGains(kp_now, set_Gains_kd)
                                    handstate = 1
                                    time.sleep(1)

                                handstate = 2
                                point[2] = point[2] + 50
                                suc, result_joint, id = sendCMD(sock, "inverseKinematic", {"targetPose": point})
                                suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": result_joint, "speed": 3})
                                while True:     #if slip is detected, increase force of grasping
                                    if (touchstate == 2):
                                        if (force_unit < 10):
                                            force_unit += 0.2
                                            if (Item[i] == 'cup' and flag == 2):
                                                containwater = True
                                        testdll.HandMotion(grap_9)
                                        kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16 = tuple(
                                            kp_unit * force_unit)
                                        kp_now = (c_double * 16)(2000, kp2, kp3, kp4, 2000, kp6, kp7, kp8, 2000, kp10,
                                                                 kp11, kp12, 4000, 2800, kp15, kp16)
                                        testdll.setGains(kp_now, set_Gains_kd)
                                        print('force:', force_unit)
                                    time.sleep(0.02)
                                    ret1, result1, id1 = sendCMD(sock, "getRobotState")  # getRobotstate
                                    if ret1:
                                        if result1 == 0 or result1 == 4:
                                            if Item[i] == 'paper':
                                                sort_flag = 4
                                            break
                                    else:
                                        print("getRobotState failed")
                                        break

                                point[2] = point[2] + 150
                                suc, result_joint, id = sendCMD(sock, "inverseKinematic", {"targetPose": point})
                                suc, result, id = sendCMD(sock, "moveByJoint",
                                                          {"targetPos": result_joint, "speed": 10})
                                wait_stop()
                                while sort_flag == 4 and Item[i] == 'paper':
                                    time.sleep(0.01)
                                    print('wait')
                                if (sort_flag == 1):
                                    Item[i] = 'plastic bag'
                                elif sort_flag == 2:
                                    Item[i] = 'napkin'
                                elif sort_flag == 3:
                                    Item[i] = 'A4'
                                print(Item[i])
                                sort_flag = 0
                                handstate = 1
                                time.sleep(0.02)
                                if (Item[i] == 'cup' or Item[i] == 'milk box'):
                                    if (containwater == True):
                                        point = [-101.3, -34, 31.4, -2.1, 161.6, -92.4, 0, 0]
                                        suc, result, id = sendCMD(sock, "moveByJoint",
                                                                  {"targetPos": point, "speed": 15})  # 移动到篮子处
                                        wait_stop()
                                        point = [-80.3, -41, 37.3, -70, 57.2, 10, 0, 0]
                                        suc, result, id = sendCMD(sock, "moveByJoint",
                                                                  {"targetPos": point, "speed": 40})  # 移动到篮子处
                                        wait_stop()
                                        force_unit = 2
                                        testdll.HandMotion(grap_9)
                                        kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16 = tuple(
                                            kp_unit * force_unit)
                                        kp_now = (c_double * 16)(2000, kp2, kp3, kp4, 2000, kp6, kp7, kp8, 2000,
                                                                 kp10, kp11, kp12, 4000, 2800, kp15, kp16)
                                        testdll.setGains(kp_now, set_Gains_kd)
                                        time.sleep(2)
                                    point = [-79, -98, 116, -113, 118, -74.8, 0, 0]
                                elif (Item[i] == 'plastic bag' or Item[i] == 'bottle'):
                                    point = [-72.4, -121.4, 132, -109, 117, -67, 0, 0]
                                elif (Item[i] == 'napkin'):
                                    point = [-86, -51, 56, -95.5, 118, -82.7, 0, 0]
                                elif (Item[i] == 'A4' or Item[i] == '纸张'):
                                    point = [-79, -98, 116, -113, 118, -74.8, 0, 0]
                                suc, result, id = sendCMD(sock, "moveByJoint",
                                                          {"targetPos": point, "speed": 35})  # 移动到篮子处
                                wait_stop()
                                testdll.HandMotion(start_pos)
                                time.sleep(0.5)
                                handstate = 0
                                point = [-37, -95, 88, -89, 94, -34, 0, 0]  # 初始化的末端法兰,关节角度
                                suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 50})
                                wait_stop()
                                break
                            else:
                                print("can not grasp ", Item[i])
                                if (i == len(Item) - 1):
                                    resultflag = 0

                        else:
                            print("can not grasp ", Item[i])
                            if (i == len(Item) - 1):
                                resultflag = 0
                else:
                    cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 49)
                    point = [-45, -88, 78.5, -56, 123, -47, 0, 0]
                    suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 20})
                    wait_stop()
                    resultflag, object_list, pos_list = check_object(cam, runtime, mat, depth, point_cloud, FLAGS, trainer)
                    point = [-37, -95, 88, -89, 94, -34, 0, 0]
                    suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 30})
                    near_pos = []
                    if (resultflag != 0):
                        print(object_list)
                        print(pos_list)
                        near_dis = 4000
                        for i in range(resultflag):
                            pos = pos_list[i]
                            if (pos[0] * pos[0] + pos[1] * pos[1] < near_dis * near_dis):
                                near_pos = pos
                                near_dis = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
                    if (near_pos != []):
                        print('near_pos:', near_pos)
                        forward_dis = near_pos[1] + 80
                        if (forward_dis > 0):
                            for i in range(round(forward_dis * 10 /100)):
                                Hold_value = master.execute(slave=17, function_code=cst.WRITE_MULTIPLE_REGISTERS,
                                                            starting_address=40022,
                                                            output_value=[-100, 0, 0])
                                time.sleep(0.1)
                    point = [-37, -95, 88, -89, 94, -34, 0, 0]  # 初始化的末端法兰,关节角度
                    suc, result, id = sendCMD(sock, "moveByJoint", {"targetPos": point, "speed": 30})
                    wait_stop()
            else:
                resultflag = 0         #nothing detected, quit
    global threadrun
    threadrun = False
    testdll.CloseCAN()
    testdll.DestroyBHandAlgorithm()
    testdll.closeCmdMemory()
    cv2.destroyAllWindows()
    cam.close()
    print("\nFINISH")


# settings of object detection
class INput:
    combine_method='nms'
    config='configs/yolov3/yolov3_garbage_coco2.yml'
    draw_threshold = 0.5
    infer_dir = None
    infer_img = 'demo/V001.png'
    match_metric = 'ios'
    match_threshold = 0.6
    opt={'use_gpu': True}
    output_dir = 'output'
    overlap_ratio = [0.25, 0.25]
    save_results = False
    slice_infer = False
    slice_size = [640, 640]
    slim_config = None
    use_vdl = False
    vdl_log_dir = 'vdl_log_dir/image'
    visualize = True

def main():
    FLAGS = INput()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    robot_ip = "192.168.1.200"
    conSuc, sock = connectETController(robot_ip)
    main()
