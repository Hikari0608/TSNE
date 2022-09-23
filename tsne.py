from openTSNE import TSNEEmbedding
from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN
from openTSNE import initialization
from openTSNE.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data import *
import demo

class test:
    def __init__(self, *args, **kwargs):
        super(test,self).__init__(*args, **kwargs)
        self.tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        self.x = np.array([])
        self.y = np.array([])
        self.cnt = 0

    def getDate(self,dir,label,shape=[256,256]):
        images = load_data(os.path.join(dir,label))
        save = []
        for d in images:
            img = demo.imread(os.path.join(data_directory,label),d)
            img = tf.image.resize(img,size=shape)
            img = tf.reshape(img,img.shape[0]*img.shape[1]*img.shape[2])
            save.append(img)

        if self.cnt == 0:
            self.x = np.array(save)
            self.y = np.array([label for _ in range(len(images))])
        else:
            self.x = np.concatenate([self.x,np.array(save)],axis=0)
            self.y = np.concatenate([self.y,np.array([label for _ in range(len(images))])],axis=0)

        self.cnt +=1
        

    def fit(self,inputs):
        embedding = self.tsne.fit(inputs)
        return embedding

    def draw2D(self,colors=demo.MOUSE_10X_COLORS,ax=None,s=10,alpha=0.4):
        return demo.plot(self.fit(self.x), self.y ,colors=colors,ax=ax,s=s,alpha=alpha)


#修改为数据集存放目录
data_directory = "C:/Users/35227/Desktop/256" 

#修改为数据集文件夹名称与对应显示色
my_dataset = {
    "Ours_90": "RED",
    "UIEB_Raw_90": "GREEN",
    "UIEB_Reference_90":"BLUE"
}

size = 10 #大小
alpha = 0.4 #透明度
title = None

if __name__ == "__main__":
    tsne = test()

    for x in my_dataset:
        tsne.getDate(dir=data_directory,label=x) #读取

    _, ax = plt.subplots(figsize=(16, 16)) #绘制
    tsne.draw2D(colors=my_dataset,ax=ax,s=size,alpha=alpha)

    if title: #额外设置
        plt.title(title)
    plt.show()
