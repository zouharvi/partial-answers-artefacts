Submitted:

|N|Architecture|Parameters|CV ACC|Test ACC|
|-|-|-|-|-|
|1|L48-relu-D0.2-L48-relu-D0.2-L48-softmax|lr 0.0015, weight decase 0.008, batch_size 64, epochs 11|76.93%|76.6%|
|2|ensemble of model 1|seq 15|76.93%|76.6%|
|3|ensemble of model 1|seq 40, epochs 10|76.93%|72.0%|
|4|ensemble of model 1|seq 20, epochs 11, fix None return|76.93%|77.0%|
|5|ensemble of model 1|seq 20, epochs 11, width 52|76.93%|76.8%|

## Experiment with otherwise `model_1` parameters 
|epochs|CV ACC|
|-|-|
|10|76.88%|
|11|76.88%|
|12|76.83%|
|13|76.76%|
|14|76.67%|