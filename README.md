# PrecisionFuturePesticide
Metformin-121_精準未來農藥

「解決方案」之內容如下:

## 1.	五分鐘介紹的youtube影片網址
https://youtu.be/fzm-K5I-LyQ

## 2.	github原始碼連結
https://github.com/chentsaimin/PrecisionFuturePesticide

## 3.	問題背景
增加農作生產率的兩個大方向為開源與節流。開源的部分包含使用肥料補充作物營養來增加其產量，而節流就是使用農藥來減少病蟲害所造成的損耗。然而農藥的使用就是一種雙面刃，它雖能解決病蟲害的問題，但同時也會殘留在作物與土壤上，對人類來說如果不慎食用過量的農藥是會對人體造成傷害，對土壤環境來說，殘留的農藥也會破壞土壤的微生態系，對之後種植的作物可能產生其他不良的影響。因此如何在一開始就能夠精準研發出除了有效以外，殘留時間又短的農藥會是一個未來的趨勢。我們會藉由人工智慧的深度學習技術，利用網路公開的農藥資料集，做出一個精準農藥預測模型，來協助篩選出有效並具有殘留時間短特性的候選農藥來進行昂貴耗時的後續試驗。

## 4.	技術架構
AI模型:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
main_input (InputLayer)      (None, 3094)              0         
_________________________________________________________________
dense_51 (Dense)             (None, 1524)              4716780   
_________________________________________________________________
batch_normalization_46 (Batc (None, 1524)              6096      
_________________________________________________________________
leaky_re_lu_46 (LeakyReLU)   (None, 1524)              0         
_________________________________________________________________
dense_52 (Dense)             (None, 768)               1171200   
_________________________________________________________________
batch_normalization_47 (Batc (None, 768)               3072      
_________________________________________________________________
leaky_re_lu_47 (LeakyReLU)   (None, 768)               0         
_________________________________________________________________
dense_53 (Dense)             (None, 384)               295296    
_________________________________________________________________
batch_normalization_48 (Batc (None, 384)               1536      
_________________________________________________________________
leaky_re_lu_48 (LeakyReLU)   (None, 384)               0         
_________________________________________________________________
dense_54 (Dense)             (None, 192)               73920     
_________________________________________________________________
batch_normalization_49 (Batc (None, 192)               768       
_________________________________________________________________
leaky_re_lu_49 (LeakyReLU)   (None, 192)               0         
_________________________________________________________________
dense_55 (Dense)             (None, 96)                18528     
_________________________________________________________________
batch_normalization_50 (Batc (None, 96)                384       
_________________________________________________________________
leaky_re_lu_50 (LeakyReLU)   (None, 96)                0         
_________________________________________________________________
dense_56 (Dense)             (None, 48)                4656      
_________________________________________________________________
batch_normalization_51 (Batc (None, 48)                192       
_________________________________________________________________
leaky_re_lu_51 (LeakyReLU)   (None, 48)                0         
_________________________________________________________________
dense_57 (Dense)             (None, 24)                1176      
_________________________________________________________________
batch_normalization_52 (Batc (None, 24)                96        
_________________________________________________________________
leaky_re_lu_52 (LeakyReLU)   (None, 24)                0         
_________________________________________________________________
dense_58 (Dense)             (None, 12)                300       
_________________________________________________________________
batch_normalization_53 (Batc (None, 12)                48        
_________________________________________________________________
leaky_re_lu_53 (LeakyReLU)   (None, 12)                0         
_________________________________________________________________
dense_59 (Dense)             (None, 6)                 78        
_________________________________________________________________
batch_normalization_54 (Batc (None, 6)                 24        
_________________________________________________________________
leaky_re_lu_54 (LeakyReLU)   (None, 6)                 0         
_________________________________________________________________
dropout_54 (Dropout)         (None, 6)                 0         
_________________________________________________________________
dense_60 (Dense)             (None, 3)                 21        
=================================================================
Total params: 6,294,171
Trainable params: 6,288,063
Non-trainable params: 6,108
_________________________________________________________________
None

## 5.	使用的資料
網路上公開的PPDB農藥資料集、台灣農委會的農藥開放資料

## 6.	開源授權方式(The MIT License):

COPYRIGHT

All contributions by Tsai-Min Chen, Chih-Han Huang:
Copyright (c) 2020 - 2020, Tsai-Min Chen, Chih-Han Huang.
All rights reserved.

All contributions by François Chollet:
Copyright (c) 2015 - 2019, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2018, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2018, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

