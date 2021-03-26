# KFC-MER
Key Facial Components Guided Micro-Expression Recognition Based on First & Second-Order Motion

This project is the implementation for our paper “Key Facial Components Guided Micro-Expression Recognition Based on First & Second-Order Motion”.
The codes need to run in the environment: Python 3.7.

1. Data preparation:

Download the input data used for our experiments (optical flow, optical strain, face segmentation probability maps of CASMEII, SMIC and SAMM database) from the following link:  https://pan.baidu.com/s/1qrGNiieM5fFj_tTlXkppYA with password ‘t1nb’ and place it under directory ‘KFC-MER\’.

It's worth noting that our codes for generating face segmentation probability maps refer to the PyTorch implementation of BiSeNet, and you can download it from https://github.com/zllrunning/face-parsing.PyTorch".

2. Model parameters:

Download the trained model parameters from the following link: https://pan.baidu.com/s/1qrGNiieM5fFj_tTlXkppYA with password ‘t1nb’ and place it under directory ‘KFC-MER\’.

3. Testing:

Run the following codes to reproduce the recognition results provided in the paper:

(1) 'KFC-MER/Code/Code_CASMEII/last_test.py'  %  CASMEII (5 classes)

(2) 'KFC-MER/Code/Code_SMIC/last_test.py'  %  SMIC (3 classes)

(3) 'KFC-MER/Code/Code_SAMM/last_test.py'  %  SAMM (5 classes)

4. Cite

If you use this code, please cite the following publication: Y.Su, J.Zhang, J.Liu, G.Zhai, "Key Facial Components Guided Micro-expression Recognition Based on First & Second-order Motion", to appear in International Conference on Multimedia and Expo (ICME2021).
