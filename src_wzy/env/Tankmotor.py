import threading
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp
import time

class Tankmotor:
    def __init__(self):
        self.master = modbus_tcp.TcpMaster(host="192.168.1.199", port=502)
        self.master.set_timeout(1.0)
        print("connected")

    def motorthread_start(self):
        motor_thread = threading.Thread(target=self.start())
        print("motor_thread_start")

    def start(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 6882, 1, output_value=1)#1是启动，2是复位（急停），3是开启单步，4是取消单步
        print("start")

    def stop(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 6882, 1, output_value=2)
        print("stop")

    def singlestep(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 6882, 1, output_value=3)
        print("singlestep")

    def cancelsinglestep(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 6882, 1, output_value=4)
        print("cancelsinglestep")

    def readstate(self):
        vel = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 8046, 1)  # 读取速度值
        error = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 9790, 1)  # 读取驱动器状态值（1是报警，0是无报警,3是复位完成，4是运动完成）
        return vel, error



if __name__ == '__main__':
    tankmotor = Tankmotor()  
    tankmotor.stop()
    #    time.sleep(100)                
    # tankmotor.start()
    #    tankmotor.singlestep()