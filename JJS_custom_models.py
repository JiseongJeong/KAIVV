import torch
import torch.nn as nn
import torchaudio.transforms as T
import time

class MK1_SimpleCNN(nn.Module):
    def __init__(self):
        super(MK1_SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #exp8에서 추가
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #exp8에서 추가
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
         #25600
        self.fc1 = nn.Linear(63488, 128)
        
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 2)  #binary

    def forward(self, x):
        x = self.conv1(x)
        #exp8에서 추가
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        #exp8에서 추가
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x



class MK1_SimpleCNN_snatch(nn.Module):
    def __init__(self):
        super(MK1_SimpleCNN_snatch, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #exp8에서 추가
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #exp8에서 추가
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
         #25600
        self.fc1 = nn.Linear(63488, 128)
        
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 2)  #binary

    def forward(self, x):
        x = self.conv1(x)
        #exp8에서 추가
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        #exp8에서 추가
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        snatch = self.fc1(x)
        x = self.relu3(snatch)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return snatch



class MultiModal_mk1(nn.Module):
    def __init__(self):
        super(MultiModal_mk1, self).__init__()

        self.bn = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU()
        self.Gelu = nn.GeLU()
        self.fc1 = nn.Linear(384, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  #binary

        checkpoint_img = torch.load('./mk1_final_asv_chkpt/mk1_final_asv_best_model_epoch_19.pt')
        self.Imagemodel = MK1_SimpleCNN_snatch().load_state_dict(checkpoint_img['model_state_dict'])

        self.Audiomodel = 오디오모델명().웨이트로딩

    def forward(self, x):   #input shape : x = {'mel' : 128사이즈 백터, 'aud' : 256사이즈 벡터}
        image_feat = self.Imagemodel(x['mel']) #(128,)
        audio_feat = self.Audiomodel(x['aud']) #(256,)

        cat_feat = torch.cat([image_feat, audio_feat])

        x = self.fc1(cat_feat)
        x = self.bn(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



#mel정보 (exp1 ~ final exp3)
#sample_rate=16000
# n_fft=400
# win_length=400
# hop_length=160
# n_mels=64 
#----------------------
# exp1 : learning rate 0.01로 고정하여 adam opim 썼을때는 val acc 0.5에 val_loss가 전혀 줄지 않았음 (늘지도 않음)  (데이터 50k)
# exp2 : learning rate 0.001로 고정하여 adam opim 썼을때는 베스트가 best_epoch, best_train_acc, best_val_acc = (8, 0.747975, 0.6784) (데이터 50k)
#       weight파일명 : first_chkpt/best_model_epoch_8.pt
#       최저로스는 23. 몇이었음...
# exp3 : exp2에서 lr을 1/2줄여보자 (데이터 50k)   -> 놉. 단 한번의 에폭도 이전 valloss를 넘어서지 못했음
#        최저로스  25.5409
# exp4 : exp2에서 lr을 1/5줄여보자 (데이터 50k) -> 처음25부터 더 안줄어서 수동중단
# exp5 : exp2에서 lr을 1/10 줄여보다 (데이터 50k) -> 처음25부터 더 안줄어서 수동중단
# exp6 : exp2에서 lr을 1/100 줄여보다 (데이터 50k) -> 처음에 24.몇이 되긴함
# exp6 : exp2에서 lr을 1/1000 줄여보자.  (데이터 50k)  -> 24.097인가가 베스트 버려...
# exp7 : exp2에서 lr을 1/10000 줄여보자.  (데이터 50k)  -> 버려

# exp8 : initial weight상태에서(exp2) 각 conv레이어 뒤에 batchnorm넣음 (데이터 50k) -> (23, 25.34713387489319, 0.6434) 더 못하는데..
        #초기웨이트 설정 탓 아닐까?
# 
# 
# finla exp1 : max balanced data를 batchnorm에 넣어봄.
## finla exp2 : bn 추가 상태에서 maximum data 넣어보기 lr 0.001 -> (27, 0.4708708925816995, 0.7666091245376079)
# final exp3 : lr 10분의1  -> (37, 0.38908260779560716, 0.8213563501849569)


#mel정보 (0913 night exp1 ~)
#sample_rate=16000
# n_fft=2048
# win_length=400
# hop_length=128
# n_mels=128 
# 128,126 사이즈 멜 이미지


# 0913 night exp1 : lr 0.001 / epoch 1~ 19 / val_loss 0.2327 / val_acc 0.9017
# 0913 night exp2 : lr 0.0001 / epoch 1 ~ 26, / val_loss 0.09167299655107962,/ val_acc 0.9678914919852034
# 0913 night exp3 : lr 0.00001 / ep ~ 29 / va valloss 0.0642  / valacc 0.9790
# 0913 night exp4 : lr 0.000001
# grad cam 적용후 중점적으로 보는 부분의 주파수?

# test셋을 노트북파일에서 계속 새로 구성해주다보니 사실 test셋이 여러번 train이나 valset에 들어갔을것.. 위 실험은 의미가 없다... 다시하는게 좋긴할텐디

# 그래서 다시함... 다시한거 어딨니?(0925)
# 데이터셋 (전체/777/42)

#4초split 데이터로 간다. 전체 데이터 먼저 드간다. 단 데이터 비율 1:1
#sr=16000, n_fft=2048, n_mels = 128, win_length=400, hop_length=512,
# img size : 128/126으로 전과동
# lr =0.001 / batch 64
# 실패!!

#동일 데이터 / 동일 transform / img 전과동'
# lr = 0.0001 / batch 64
# ㅇ? 1에폭 val_acc이 92퍼? 2에폭에서 좀 떨어지는군 1퍼정도




class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)


        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1)
        )


        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2)
        )


        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class JGoogLeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(JGoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        #self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=(1, 0))  # 128x126 input에 맞게하기 위해..

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# jgooglenet
## test1
### 아키텍쳐 수정 : conv1 input channel 3 -> 1 / output channel 64 -> 32
### 아키텍쳐 수정 : conv2  in/out (64,64) -> (32,64)
### 아키텍쳐 수정 : conv3 padding = (1,0)
### 데이터 : 전체 데이터(seed=777/42)  
### lr : 0.0001
### patience : 3
### save file name as : f'jgooglenet_best_model_epoch{title_numbering}_{epoch}.pt'
### result : test_accuracy :  0.8069638982047741 	 test_f1 :  0.8069632896770849 	 test_auc :  0.8070401172311723 	 test_recall :  0.7982593389399569
### result log plot : /home/alpaco/mel_spec_approach/training_log_plot.jpg
### best model saved as : /home/alpaco/mel_spec_approach/second_chkpt/jgooglenet_best_model_epoch_6.pt

## test2
### 아키텍쳐 / 데이터 그대로
### lr 0.00001

## test3 
### 다 그대로
### lr 0.000001

## test 4
### 다 그대로
### lr 0.0000001
### 최저 loss 갱신 : second_chkpt/jgooglenet_back_best_model_epoch_lr1e-07_7.pt

## test 5
### lr 0.00000001
#### 훈련 종료! test4가 베스트인듯


# batch norm 추가한 inception
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.BatchNorm2d(red5x5),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out5x5)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class JBNGoogLeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(JBNGoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=(1, 0)), 
            nn.BatchNorm2d(192)
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# jbngooglenet
## test1
### lr : 0.005
### 데이터 : 전체 데이터(seed=777/42)  
### patience : 3
### 학습 안되고 실패

## test2
### lr : 0.0001  이전 모델 실험때와  lr 통일해줌
### ~
### lr = 0.0000001 까지 내리다가의 test 결과는 (jbngooglenet_best_model_epoch_5.pt')
### test_accuracy :  0.6290688498717696 	 test_f1 :  0.5994341605939408 	 test_auc :  0.6267550862935469 	 test_recall :  0.8933111676119695




class MK2_1DCNN(nn.Module):
    
    def __init__(self):
        super(MK2_1DCNN, self).__init__()
        self.conv_stem = nn.Conv1d(49, 49, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv_head1 = nn.Sequential(
            nn.Conv1d(49, 7, kernel_size=1, padding=0), 
            nn.BatchNorm1d(7)
        )
        self.conv_head2 = nn.Sequential(
            nn.Conv1d(7, 1, kernel_size=1, padding=0), 
            nn.BatchNorm1d(1)
        )

        self.fc1 = nn.Linear(768, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  #binary

    def forward(self, x):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = x


        # x1 = x[:][0]
        # x2 = x[:][1]
        # x3 = x[:][2]
        # x4 = x[:][3]
        # x5 = x[:][4]
        # x6 = x[:][5]
        # x7 = x[:][6]
        # x8 = x[:][7]
        # x9 = x[:][8]
        # x10 = x[:][9]
        # x11 = x[:][10]
        # x12 = x[:][11]
        # x13 = x[:][12]

        # x1 = x[0]
        # x2 = x[1]
        # x3 = x[2]
        # x4 = x[3]
        # x5 = x[4]
        # x6 = x[5]
        # x7 = x[6]
        # x8 = x[7]
        # x9 = x[8]
        # x10 = x[9]
        # x11 = x[10]
        # x12 = x[11]
        # x13 = x[12]

        x1 = self.conv_stem(x1)
        x2 = torch.add(x1, x2)

        x2 = self.conv_stem(x2)
        x3 = torch.add(x2, x3)

        x3 = self.conv_stem(x3)
        x4 = torch.add(x3, x4)

        x4 = self.conv_stem(x4)
        x5 = torch.add(x4, x5)

        x5 = self.conv_stem(x5)
        x6 = torch.add(x5, x6)

        x6 = self.conv_stem(x6)
        x7 = torch.add(x6, x7)

        x7 = self.conv_stem(x7)
        x8 = torch.add(x7, x8)

        x8 = self.conv_stem(x8)
        x9 = torch.add(x8, x9)

        x9 = self.conv_stem(x9)
        x10 = torch.add(x9, x10)

        x10 = self.conv_stem(x10)
        x11 = torch.add(x10, x11)

        x11 = self.conv_stem(x11)
        x12 = torch.add(x11, x12)

        x12 = self.conv_stem(x12)
        x13 = torch.add(x12, x13)

        conv_head1 = self.conv_head1(x13)
        conv_head2 = self.conv_head2(conv_head1)

        flat_feature = conv_head2.view(conv_head2.size(0), -1)

        x = self.dropout(flat_feature)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

class test_MK2_1DCNN(nn.Module):
    def __init__(self):
        super(test_MK2_1DCNN, self).__init__()

        self.conv_stems = nn.ModuleList([
            nn.Conv1d(49, 49, kernel_size=1, padding=0) for _ in range(12)
        ])
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv_head1 = nn.Sequential(
            nn.Conv1d(49, 7, kernel_size=1, padding=0),
            nn.BatchNorm1d(7),
            nn.ReLU()  
        )
        self.conv_head2 = nn.Sequential(
            nn.Conv1d(7, 1, kernel_size=1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU()  # Adding ReLU after the convolutional layer
        )

        self.fc1 = nn.Linear(768, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  # binary

    def forward(self, x):
        start = time.time()
        processed_elements = []
     
        x = x.view(-1, 13, 49, 768)
   
        #를 수정(수정번호1)
        # for i in range(12):
        #     xi = self.conv_stems[i](x[:, i, :, :]) 
            
        #     xi = self.relu(xi)   
        #     processed_elements.append(xi)

        #로 수정(수정번호1)
        for i in range(12):
            processed_elements.append(x[:, i, :, :])      

        #를 수정(수정번호1)
        # for i in range(1, 12):
        #     processed_elements[i] = self.conv_stems[i](processed_elements[i])
        #     processed_elements[i] = self.relu(processed_elements[i])  
        #     processed_elements[i] = torch.add(processed_elements[i], processed_elements[i - 1])

        #로 수정
        for i in range(0, 12):
            processed_elements[i] = self.conv_stems[i](processed_elements[i])
            processed_elements[i] = self.relu(processed_elements[i])
            if i > 0:
                processed_elements[i] = torch.add(processed_elements[i], processed_elements[i - 1])

            

        conv_head1 = self.conv_head1(processed_elements[-1])
        conv_head2 = self.conv_head2(conv_head1)

        flat_feature = conv_head2.view(conv_head2.size(0), -1)

        x = self.dropout(flat_feature)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

## 배치 128, whole data, lr 0.001
### 개느리다. 한 에폭에 2시간 넘음ㅠㅠ

## 배치 256, 20000data(10%) lr 0.001
### Epoch 1 학습 결과는!!! Validation Loss = 0.6435, Validation Accuracy = 0.6445 에서 끝


### interval 메모

# 시작!: 0.00004 sec
# gpu 사용 가능 여부 True
# Epoch 1 (Training):   0%|          | 0/8 [00:00<?, ?it/s]
# Epoch 1 (Training):  25%|██▌       | 2/8 [00:00<00:00, 13.27it/s]
# 1번째 데이터 토치오디오 로딩 완료!:  0.00126 sec
# 1번째 휴버트 특성 추출 완료!:  0.02059 sec
# 1번째 특성 스택킹완료!:  0.02071 sec
# 베치데이터 gpu 태우기: 0.02535 sec
# 라벨데이터 gpu 태우기 : 0.02540 sec
# 베치데이터 모델에 넣고 아웃풋 받기 : 0.04618 sec
# 라벨과 비교(criterion) : 0.04643 sec
# 역전파 완료 : 0.08928 sec
# 옵티마이지 ㅇ완료료 : 0.09268 sec
# train_loss 계산 완료 : 0.09278 sec
# 6번째 데이터 토치오디오 로딩 완료!:  0.00074 sec
# 6번째 휴버트 특성 추출 완료!:  0.01758 sec
# 6번째 특성 스택킹완료!:  0.01768 sec
# 베치데이터 gpu 태우기: 0.11127 sec
# 라벨데이터 gpu 태우기 : 0.11133 sec
# 베치데이터 모델에 넣고 아웃풋 받기 : 0.11516 sec
# 라벨과 비교(criterion) : 0.11527 sec
# 역전파 완료 : 0.15053 sec
# 옵티마이지 ㅇ완료료 : 0.15324 sec
# train_loss 계산 완료 : 0.15333 sec
# 0번째 데이터 토치오디오 로딩 완료!:  0.00069 sec
# 0번째 휴버트 특성 추출 완료!:  0.01766 sec
# 0번째 특성 스택킹완료!:  0.01776 sec
# 베치데이터 gpu 태우기: 0.17398 sec
# 라벨데이터 gpu 태우기 : 0.17404 sec
# 베치데이터 모델에 넣고 아웃풋 받기 : 0.17909 sec
# 라벨과 비교(criterion) : 0.17922 sec

### 전체데이터셋에 아키택쳐 수정후 batch를 32로하니
#### Epoch 1 (Training):   3%|▎         | 148/5069 [03:08<1:44:35,  1.28s/it]

