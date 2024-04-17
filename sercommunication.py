import serial
import serial.tools.list_ports

class Sercommunication():

    def __init__(self, com, baudrate, timeout):
        self.port = com
        self.bps = baudrate
        self.timeout = timeout
        global Ret
        try:
            self.ser = serial.Serial(self.port, self.bps, timeout=self.timeout)
            if self.ser.is_open:
                Ret = True
                self.ser.close()
        except Exception as e:
            print("---异常---：", e)

    def Open_Ser(self):
        self.ser.open()
        print("串口打开")

    def Close_Ser(self):
        self.ser.close()
        print(self.ser.is_open)  # 检验串口是否打开


    def Read_Size(self, size):
         return self.ser.read(size=size)

    def Read_Line(self):
        return self.ser.readline()


    def Send_data(self, data):
        self.ser.write(data)

    def Recive_data(self,way):

        print("开始接收数据：")
        while True:
            try:
                if self.ser.in_waiting:
                    if(way == 0):
                        for i in range(self.ser.in_waiting):
                            print("接收ascii数据："+str(self.Read_Size(1)))
                            data1 = self.Read_Size(1).hex()
                            data2 = int(data1,16)
                            print("收到数据十六进制："+data1+"  收到数据十进制："+str(data2))
                    if(way == 1):
                        data = self.ser.read_all().decode('uft-8')
                        print("接收ascii数据：", data)
            except Exception as e:
                print("异常报错：",e)
