import os
import torch
import torchvision
import cv2
import numpy as np
import argparse
import sys
sys.path.append('.')

from models.humaniflow_demo_model import HumaniflowModel
from models.smpl import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector
from configs.humaniflow_config import get_humaniflow_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths
from predict.predict_demoHumaniflow import predict_humaniflow
import socket
from _thread import *
import time  # time debug

humaniflow_model = None
humaniflow_cfg = None
hrnet_model = None
pose2D_hrnet_cfg = None
edge_detect_model = None
object_detect_model = None
smpl_model = None
device = None
joints2Dvisib_threshold = None
num_pred_samples = None

def threadedConnection(clientSocket, addr):
    start = time.time()
    finalVal = bytearray()
    connection = True
    recvData = True
    datSize = 0
    dataSize = 0
    print("Connected by", addr)

    # client가 접속을 끊을 때까지 반복
    while connection:
        while recvData:
            try:
                if(datSize == 0):
                    datSize = clientSocket.recv(4)
                    if not datSize:
                        print("Disconnected by", addr[0], ':', addr[1])
                        recvData = False
                        break
                    dataSize = int.from_bytes(datSize, "little")
                    recvData = True

                data = clientSocket.recv(dataSize)

                finalVal = finalVal + data
                #print("Receiving actual data : ", len(data))
                #print("FinalVal : ", len(finalVal), " / ", dataSize)

                if len(finalVal) == dataSize:
                    recvData = False
            except ConnectionResetError as e:
                print("Error : Disconnected by", addr[0], ':', addr[1])
                recvData = False

        global device
        #process received image
        encoded_image = np.asarray(finalVal, dtype="uint8")
        original_Img = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        #original_Img = cv2.cvtColor(cv2.imdecode(encoded_image, cv2.COLOR_BGR2RGB))
        #encoded_image = np.asarray(givenImg, dtype="uint8")
        original_Img = torch.from_numpy(original_Img.transpose(2, 0, 1)).float().to(device) / 255.0
        #original_Img = original_Img.astype(np.float32)

        try:
            # ------------------------- Predict -------------------------\
            global joints2Dvisib_threshold, num_pred_samples
            torch.manual_seed(0)
            np.random.seed(0)
            #TODO : predict_humaniflow should return array of shapes
            out = predict_humaniflow(humaniflow_model=humaniflow_model,
                               humaniflow_cfg=humaniflow_cfg,
                               hrnet_model=hrnet_model,
                               hrnet_cfg=pose2D_hrnet_cfg,
                               edge_detect_model=edge_detect_model,
                               device=device,
                               image=original_Img,
                               object_detect_model=object_detect_model,
                               num_pred_samples=num_pred_samples,
                               joints2Dvisib_threshold=joints2Dvisib_threshold)
            clientSocket.send(np.array(out).tobytes())
            connection = False
        except ConnectionResetError as e:
            print("Error : Disconnected by", addr[0], ':', addr[1])
            connection = False

    clientSocket.close()
    end = time.time()
    print(time.ctime(start), ":", f"{end - start:.5f} sec >>> full connection done")

def main():

    # ------------------------- Arg Parsing -------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--humaniflow_weights', '-W3D', type=str, default='./model_files/humaniflow_weights.tar')
    parser.add_argument('--humaniflow_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='./model_files/pose_hrnet_w48_384x288.pth')

    parser.add_argument('--gender', '-G', type=str, default='neutral', choices=['neutral', 'male', 'female'],
                        help='Gendered SMPL models may be used.')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--num_pred_samples', '-NP', type=int, default=50)

    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    if args.gender != 'neutral':
        raise NotImplementedError

    # ------------------------- Model Loading -------------------------
    start = time.time()  # time debug
    global pose2D_hrnet_cfg, humaniflow_cfg, object_detect_model, hrnet_model, edge_detect_model, smpl_model, humaniflow_model, joints2Dvisib_threshold, num_pred_samples

    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    humaniflow_cfg = get_humaniflow_cfg_defaults()
    if args.humaniflow_cfg is not None:
        humaniflow_cfg.merge_from_file(args.humaniflow_cfg)
        print('\nLoaded HuManiFlow config from', args.humaniflow_cfg)
    else:
        print('\nUsing default HuManiFlow config.')

    # Bounding box / Object detection model
    object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)

    if object_detect_model is not None:
        object_detect_model.eval()

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(args.pose2D_hrnet_weights, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', args.pose2D_hrnet_weights)

    hrnet_model.eval()

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=humaniflow_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=humaniflow_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=humaniflow_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=humaniflow_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    print(
        '\nUsing {} SMPL model with {} shape parameters.'.format(args.gender, str(humaniflow_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender=args.gender,
                      num_betas=humaniflow_cfg.MODEL.NUM_SMPL_BETAS).to(device)

    # HuManiFlow - 3D shape and pose distribution predictor
    humaniflow_model = HumaniflowModel(device=device,
                                       model_cfg=humaniflow_cfg.MODEL,
                                       smpl_parents=smpl_model.parents.tolist()).to(device)
    checkpoint = torch.load(args.humaniflow_weights, map_location=device)
    humaniflow_model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
    humaniflow_model.pose_so3flow_transform_modules.eval()
    print('\nLoaded HuManiFlow weights from', args.humaniflow_weights)

    humaniflow_model.eval()

    joints2Dvisib_threshold = args.joints2Dvisib_threshold
    num_pred_samples = args.num_pred_samples

    end = time.time()  # time debug
    print(f"{end - start:.5f} sec >>>>>>>>>>>>> model Loading")

    # ------------------------- Socket Connection -------------------------

    host =''
    port =''

    # 서버 시작
    # AF_INET이 IPv4 / 소켓 타입은 TCP
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # port를 사용중이라는 에러 해결을 위해서 사용
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 소켓을 특정 network interface & port num에 연결하는데 사용되는 것이 bind함수
    # Host는 hostname, ip address, 모든 network interface로부터의 접근을 허용하는 ""중 하나를 사용가능.
    serverSocket.bind((host, port))

    # 서버가 클라이언트의 접속을 허용하도록 한다
    serverSocket.listen()

    # Debug
    print("server start")

    # client 접속시 accept 함수에서 새로운 소켓을 return.
    # 새로운 쓰레드에서 해당 소켓을 써서 통신을 진행.
    while True:
        # Debug
        print("Wait")

        clientSocket, addr = serverSocket.accept()
        # clientSocket.settimeout(4.0)

        start_new_thread(threadedConnection, (clientSocket, addr))

    #if you change the code for while to have some kind of way to get out of the loop, this should be used.
    #serverSocket.close()

if __name__ == '__main__':
    sys.exit(main())