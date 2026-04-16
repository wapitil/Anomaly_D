
if __name__=="__main__":
    """ 
    - RDK端：
        如果检测到 新布来了 信号
        摄像头采集一百张照片传回服务器等待新模型训练
    - PC端：
        如果在 /home/wapiti/Projects/Anomaly_D/Datasets/Server/images 路径下检测到新文件夹
        trian_AD() # 开启异常检测模型训练
        模型训练好后，存放至 /home/wapiti/Projects/Anomaly_D/Datasets/Server/models 并创建新文件夹
        将模型推送至RDK端
    - RDK端：
        检测 RDK 本地的 models 文件 是否有新模型更新
        如果有新异常检测模型，则更换模型
    """
    # 

