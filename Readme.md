##### CRNN网络结构：
![crnn网络结构：CNN+Bi-RNN+CTC_loss](https://i.loli.net/2018/10/25/5bd15c9aabc36.png
)

##### 输入图片：
![输入图片示例（大小：w*h=192*32）](https://i.loli.net/2018/10/25/5bd15d3e8b4de.jpg
)

                                   输入：[batch_size,192,32,1]->

               Conv1-bn1-maxpool1->      [batch_size,96,16, 32]

               Conv2-bn2-maxpool2->      [batch_size,48,8, 64]

               Conv3-bn3-maxpool3->      [batch_size,24,4,128]

               Conv4-bn4-maxpool4->      [batch_size,12,2,256]

               Conv5-bn5->               [batch_size,11,1,512]

               Reshape->                 [batch_size,512, 11]

               Bi-lstm->                 [batch_size,512, 20]

               Output Sequence->         [512,batch_size,993]

               Ctc_loss

***
### 实验：

#### 训练样本大小：
10072 中文字符数量：992个汉字+1个空格=993

#### 超参数：

Batch_size: 8

Step:1259(1259*8 = 10072)

Epoch: 10

Learning_rate:0.01
#### 实验结果：
![CTC_LOSS](https://i.loli.net/2018/10/29/5bd6ba34aae21.png
)
![训练结果](https://i.loli.net/2018/10/29/5bd6ba3d8c508.png
)

迭代次数太少了，最后的测试精度还不够高。

简书：[利用CRNN来识别图片中的文字](https://www.jianshu.com/p/085c2f6ab886)
