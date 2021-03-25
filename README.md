# KFC-MER
This project is the implementation for our paper“Key Facial Components Guided Micro-Expression Recognition Based on First & Second-Order Motion”accepted by ICME2021.
The codes need to run in the environment: Python 3.7.
Data preparation:
Download the input data used for our experiments (optical flow, optical strain, face segmentation probability maps of CASMEII, SMIC and SAMM database) from the following link:  https://pan.baidu.com/s/1qrGNiieM5fFj_tTlXkppYA with password‘t1nb’ and place it under directory ‘KFC-MER\’
Model parameters:
Download the trained model parameters from the following link: https://pan.baidu.com/s/1qrGNiieM5fFj_tTlXkppYA with password‘t1nb’ and place it under directory ‘KFC-MER\’
Testing:
Run the following codes to reproduce the recognition results provided in the paper:
(1) 'KFC-MER/Code/Code_CASMEII/last_test.py'  %  CASMEII (5 classes)
(2) 'KFC-MER/Code/Code_SMIC/last_test.py'  %  SMIC (3 classes)
(3) 'KFC-MER/Code/Code_SAMM/last_test.py'  %  SAMM (5 classes)

Cite
If you use this code, please cite the following publication: J.Liu, J.Zhang, G.Zhai, Y.Su, "Key Facial Components Guided Micro-Expression Recognition Based on First & Second-Order Motion", to appear in International Conference on Multimedia and Expo(ICME2021).
