import os

from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# 初始化下载器，LocalDirectory 换成你想保存的实际路径
local_directory = os.getenv("SOCCERNET_VISION_ROOT", "/path/to/caption-2024")
mySNdl = SNdl(LocalDirectory=local_directory)

# 开始下载（包含训练集、验证集、测试集和 2024 挑战集）
mySNdl.downloadDataTask(task="caption-2024", split=["train", "valid", "test", "challenge"])
