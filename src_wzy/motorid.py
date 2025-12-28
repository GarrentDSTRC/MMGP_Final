from typing import OrderedDict
import serial
# import nidaqmx
import time
import math
import os
import datetime
import numpy as np

# desktop_dir = 'C:/Users/test/Desktop/data0626/bar'
# os.makedirs(desktop_dir, exist_ok=True)

# =========================motor control============================
buffer = [0x01, 0x10, 0x00, 0xAC, 0x00, 0x01, 0x02, 0x00, 0x00]

DeepSea_SERVO_FRAME_HEADER = 0xff
DeepSea_SERVO_POS_TIME_WRITE = 0x83
DeepSea_SERVO_POS_TIME_WRITE_MemAddr = 42
DeepSea_SERVO_READ = 2
DeepSea_SERVO_READ_POS_MemAddr = 56
DeepSea_SERVO_READ_TIME_MemAddr = 58
DeepSea_SERVO_ID_MemAddr = 5
DeepSea_SERVO_MIN_ANGLE_LIMIT = 9
DeepSea_SERVO_MAX_ANGLE_LIMIT = 11  # 0x10 4096
DeepSea_SERVO_MAX_TORQUE_LIMIT = 16  # 0x03 800
One_date_length = 4
Total_data_length = 14 # (one_date_length+1)*number_of_motor+4
def GET_LOW_BYTE(A):
    return A & 0xFF

def GET_HIGH_BYTE(A):
    return (A >> 8) & 0xFF

def BYTE_TO_HW(A, B):
    return ((A << 8) | B) & 0xFFFF

def DeepSeaCheckSum(buf, length):
    temp = 0
    for i in range(2, length - 1):
        temp += buf[i]
    temp = ~temp
    return temp & 0xFF

def DeepSeaSerialServoMoveall(ser, id, position, time):
    buf = bytearray(18)
    buf[0] = buf[1] = DeepSea_SERVO_FRAME_HEADER
    buf[2] = 0xfe
    buf[3] = Total_data_length
    buf[4] = DeepSea_SERVO_POS_TIME_WRITE
    buf[5] = DeepSea_SERVO_POS_TIME_WRITE_MemAddr
    buf[6] = One_date_length

    buf[7] = id[0]
    buf[8] = GET_HIGH_BYTE(position[0])
    buf[9] = GET_LOW_BYTE(position[0])
    buf[10] = GET_HIGH_BYTE(time)
    buf[11] = GET_LOW_BYTE(time)

    buf[12] = id[1]
    buf[13] = GET_HIGH_BYTE(position[1])
    buf[14] = GET_LOW_BYTE(position[1])
    buf[15] = GET_HIGH_BYTE(time)
    buf[16] = GET_LOW_BYTE(time)

    buf[17] = id[2]
    buf[18] = GET_HIGH_BYTE(position[2])
    buf[19] = GET_LOW_BYTE(position[2])
    buf[20] = GET_HIGH_BYTE(time)
    buf[21] = GET_LOW_BYTE(time)


    buf[17] = DeepSeaCheckSum(buf, 18)
    ser.write(buf)


def DeepSeaSerialServoMove(ser, id, position, time):
    buf = bytearray(11)
    buf[0] = buf[1] = DeepSea_SERVO_FRAME_HEADER
    buf[2] = id
    buf[3] = 7
    buf[4] = 3
    buf[5] = DeepSea_SERVO_POS_TIME_WRITE_MemAddr
    buf[6] = GET_HIGH_BYTE(position)
    buf[7] = GET_LOW_BYTE(position)
    buf[8] = GET_HIGH_BYTE(time)
    buf[9] = GET_LOW_BYTE(time)
    buf[10] = DeepSeaCheckSum(buf, 11)
    ser.write(buf)



def DeepSeaReceiveHandle(ser, ret):
    frameStarted = False
    receiveFinished = False
    frameCount = 0
    dataCount = 0
    dataLength = 2
    recvBuf = bytearray(32)
    while ser.inWaiting():
        rxBuf = ser.read(ser.inWaiting())
        if len(rxBuf) >= 6:
            ret[0] = rxBuf[5]
        if len(rxBuf) >= 7:
            ret[1] = rxBuf[6]

# def DeepSeaServoReadPosition(ser,id):
#     buf = bytearray(8)
#     data = bytearray(2)
#     buf[0] = buf[1] = DeepSea_SERVO_FRAME_HEADER
#     buf[2] = id
#     buf[3] = 4
#     buf[4] = DeepSea_SERVO_READ
#     buf[5] = DeepSea_SERVO_READ_POS_MemAddr
#     buf[6] = 2
#     buf[7] = DeepSeaCheckSum(buf, 8)
#     ser.write(buf)
#     DeepSeaReceiveHandle(ser, data)
#     ang = BYTE_TO_HW(data[0], data[1])
#     ang1 = ang * 360 / 4096
#     return ang1


def DeepSeaReceiveHandle1(ser, ret):
        frameStarted = False
        receiveFinished = False
        frameCount = 0
        dataCount = 0
        dataLength = 2
        recvBuf = bytearray(32)
        k=ser.inWaiting()
        #print(k)
        while ser.inWaiting():
            rxBuf = ser.read(ser.inWaiting())
            #print(rxBuf)
            if len(rxBuf) >= 6:
                ret[0] = rxBuf[5]
            if len(rxBuf) >= 7:
                ret[1] = rxBuf[6]

def DeepSeaServoReadPosition(ser, id):
        bufff = bytearray(8)
        data_angle = bytearray(2)

        # 设置通信协议的帧头、伺服ID、长度等信息
        bufff[0] = bufff[1] = DeepSea_SERVO_FRAME_HEADER
        bufff[2] = id
        bufff[3] = 4  # 数据长度
        bufff[4] = DeepSea_SERVO_READ  # 读取指令
        bufff[5] = 0x38  # 要读取的地址
        bufff[6] = 0x02  # 要读取的数据长度

        # 计算并设置校验和
        bufff[7] = DeepSeaCheckSum(bufff, 8)

        # 发送指令包
        ser.write(bufff)
        # print(bufff)  # 打印指令包以供调试
        time.sleep(0.0001)
        # 接收并处理返回的数据
        DeepSeaReceiveHandle1(ser,data_angle)
        ang = BYTE_TO_HW(data_angle[0], data_angle[1])
        ang1 = ang * 360 / 4096
        return ang1

def ServoMoveall(ID, angle1,angle2,angle3):
    dpos = [int(angle1 * 4096 / 360), int(angle2 * 4096 / 360), int(angle3 * 4096 / 360)]
    # dpos = int(angle * 4096 / 360)
    
    DeepSeaSerialServoMoveall(ser, ID, dpos, 0)

def ServoMove(ID, angle):
    # dpos = [int(angle1 * 4096 / 360), int(angle2 * 4096 / 360)]
    dpos = int(angle * 4096 / 360)
    
    DeepSeaSerialServoMove(ser, ID, dpos, 0)

def ReadPos(ID):
    PosF = DeepSeaServoReadPosition(ser, ID)
    return PosF

def ServoSetID(newID):
    buf = bytearray(8)
    buf[0] = buf[1] = DeepSea_SERVO_FRAME_HEADER
    buf[2] = 0xfe
    buf[3] = 4
    buf[4] = 0x03
    buf[5] = DeepSea_SERVO_ID_MemAddr
    buf[6] = newID
    buf[7] = DeepSeaCheckSum(buf, 8)
    ser.write(buf)



if __name__ == '__main__':

    ser = serial.Serial("COM5", 115200)    # open the serial, baudrate is 115200
    if ser.isOpen():                        # Check whether successfull or not
        print("Opened the port %s successfully."%ser.name)
    else:
        print("Failed to open the port.") 


    # ServoMove(7,180)
    # time.sleep(0.1)
    # ServoMove(8,180)
    # time.sleep(0.1)
    # ServoMove(23,170)
    # time.sleep(0.01)
    # pos = DeepSeaServoReadPosition(ser,23)
    # print(pos)
    #ServoSetID(9)
    theta0=38.386
    ServoMove(1,(180+theta0))
    time.sleep(1)
    ServoMove(1,(180-2*theta0))
    # time.sleep(1)
    # ServoMove(17,175)
    # time.sleep(1)
    # ServoMove(18,183)


    # omega = 2 * math.pi*1.3
    # offset = 180
    # amplitude = 20
    # while 1:
    #     t = time.time()
    #     theta1 = round(offset + amplitude * math.sin(omega * t))
    #     theta2 = round(offset + amplitude * math.sin(omega * t))
        
        
    #     ServoMove(12,theta1)
    #     print(theta1)
    #     time.sleep(0.001)
    #     pos = DeepSeaServoReadPosition(ser,12)
    #     print(pos)

        # ServoMoveall([7,8,9], theta1,theta1,theta1)
        # time.sleep(0.01)
        # ServoMove(2,theta1)
        # # time.sleep(0.01)
        # ServoMove(3,theta1)
        # time.sleep(0.01)

        

    
    

