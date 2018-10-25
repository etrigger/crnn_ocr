# crnn_ocr
Using crnn to recognize Chinese characters in picture。

网络结构：
输入：	  
      [batch_size,192,32,1]->
      Conv1-bn1-maxpool1->	[batch_size,96,16, 32]
		  Conv2-bn2-maxpool2->	[batch_size,48, 8, 64]
		  Conv3-bn3-maxpool3->	[batch_size,24, 4,128]
		  Conv4-bn4-maxpool4->	[batch_size,12, 2,256]
		  Conv5-bn5->				[batch_size,11, 1,512]
		  Reshape->				[batch_size, 512, 11]
		  Bi-lstm->					[batch_size, 512, 20]
		  Output Sequence->		[512, batch_size,993]
		  Ctc_loss
