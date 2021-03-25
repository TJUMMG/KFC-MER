from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch
import xlrd
import os


apex_parsing_map_data_path = '../../Dataset/SMIC/SMIC_apex_seg_prob_map_augment'
OF_data_path = '../../Dataset/SMIC/SMIC_onset_apex_OF_augment'
OS_data_path = '../../Dataset/SMIC/SMIC_onset_apex_OS_augment'


class Dataload(data.Dataset):
    def __init__(self, split, leave_out, transform=None):
        self.transform = transform
        self.split = split

        if self.split == 'Training':
            self.train_data = []
            self.train_labels = []

            subfolders = os.listdir(apex_parsing_map_data_path)
            for sub_folder in subfolders:
                if sub_folder != leave_out:
                    apex_parsing_map_subfolder_path = os.path.join(apex_parsing_map_data_path, sub_folder)
                    classes = os.listdir(apex_parsing_map_subfolder_path)
                    for cls in classes:
                        video_folders = os.listdir(os.path.join(apex_parsing_map_subfolder_path, cls))
                        for video_name in video_folders:
                            apex_parsing_map_video_path = os.path.join(apex_parsing_map_subfolder_path, cls, video_name)
                            OF_video_path = os.path.join(OF_data_path, sub_folder, cls, video_name)
                            OS_video_path = os.path.join(OS_data_path, sub_folder, cls, video_name)

                            if cls == 'negative':
                                train_label = 0
                            elif cls == 'positive':
                                train_label = 1
                            elif cls == 'surprise':
                                train_label = 2
                            else:
                                print('A wrong sample has been selected.')

                            OF_img_list = os.listdir(OF_video_path)
                            for OF_img_name in OF_img_list:
                                if OF_img_name[0] == 'u':
                                    u_img_path = os.path.join(OF_video_path, OF_img_name)
                                    v_img_path = os.path.join(OF_video_path, 'v' + OF_img_name[1:])
                                    m_img_path = os.path.join(OF_video_path, 'm' + OF_img_name[1:])
                                    img_u = Image.open(u_img_path)
                                    img_u = img_u.resize((224, 224))
                                    img_u = np.array(img_u)
                                    img_v = Image.open(v_img_path)
                                    img_v = img_v.resize((224, 224))
                                    img_v = np.array(img_v)
                                    img_m = Image.open(m_img_path)
                                    img_m = img_m.resize((224, 224))
                                    img_m = np.array(img_m)
                                    img_u = img_u[:, :, np.newaxis]
                                    img_v = img_v[:, :, np.newaxis]
                                    img_m = img_m[:, :, np.newaxis]
                                    OF_img = np.concatenate((img_u, img_v, img_m), axis=2)

                                    exx_img_path = os.path.join(OS_video_path, 'exx' + OF_img_name[1:])
                                    eyy_img_path = os.path.join(OS_video_path, 'eyy' + OF_img_name[1:])
                                    exy_img_path = os.path.join(OS_video_path, 'exy' + OF_img_name[1:])
                                    img_exx = Image.open(exx_img_path)
                                    img_exx = img_exx.resize((224, 224))
                                    img_exx = np.array(img_exx)
                                    img_eyy = Image.open(eyy_img_path)
                                    img_eyy = img_eyy.resize((224, 224))
                                    img_eyy = np.array(img_eyy)
                                    img_exy = Image.open(exy_img_path)
                                    img_exy = img_exy.resize((224, 224))
                                    img_exy = np.array(img_exy)
                                    img_exx = img_exx[:, :, np.newaxis]
                                    img_eyy = img_eyy[:, :, np.newaxis]
                                    img_exy = img_exy[:, :, np.newaxis]
                                    OS_img = np.concatenate((img_exx, img_eyy, img_exy), axis=2)

                                    img = np.concatenate((OF_img, OS_img), axis=2)

                                    selected_channels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
                                    for idx in range(0, len(selected_channels)):
                                        apex_parsing_map_img_path = os.path.join(apex_parsing_map_video_path, 'apex_' + str(selected_channels[idx]) + '_segprob' + OF_img_name[1:])
                                        apex_parsing_map_img = Image.open(apex_parsing_map_img_path)
                                        apex_parsing_map_img = apex_parsing_map_img.resize((224, 224))
                                        apex_parsing_map_img = np.array(apex_parsing_map_img)
                                        apex_parsing_map_img = apex_parsing_map_img[:, :, np.newaxis]

                                        img = np.concatenate((img, apex_parsing_map_img), axis=2)
                                    print(img.shape)

                                    self.train_data.append(img)
                                    self.train_labels.append(train_label)


        if self.split == 'Testing':
            self.test_data = []
            self.test_labels = []

            subfolders = os.listdir(apex_parsing_map_data_path)
            for sub_folder in subfolders:
                if sub_folder == leave_out:
                    apex_parsing_map_subfolder_path = os.path.join(apex_parsing_map_data_path, sub_folder)
                    classes = os.listdir(apex_parsing_map_subfolder_path)
                    for cls in classes:
                        video_folders = os.listdir(os.path.join(apex_parsing_map_subfolder_path, cls))
                        for video_name in video_folders:
                            apex_parsing_map_video_path = os.path.join(apex_parsing_map_subfolder_path, cls, video_name)
                            OF_video_path = os.path.join(OF_data_path, sub_folder, cls, video_name)
                            OS_video_path = os.path.join(OS_data_path, sub_folder, cls, video_name)

                            if cls == 'negative':
                                test_label = 0
                            elif cls == 'positive':
                                test_label = 1
                            elif cls == 'surprise':
                                test_label = 2
                            else:
                                print('A wrong sample has been selected.')

                            OF_img_list = os.listdir(OF_video_path)
                            for OF_img_name in OF_img_list:
                                if OF_img_name == 'u.png':
                                    u_img_path = os.path.join(OF_video_path, OF_img_name)
                                    v_img_path = os.path.join(OF_video_path, 'v.png')
                                    m_img_path = os.path.join(OF_video_path, 'm.png')
                                    img_u = Image.open(u_img_path)
                                    img_u = img_u.resize((224, 224))
                                    img_u = np.array(img_u)
                                    img_v = Image.open(v_img_path)
                                    img_v = img_v.resize((224, 224))
                                    img_v = np.array(img_v)
                                    img_m = Image.open(m_img_path)
                                    img_m = img_m.resize((224, 224))
                                    img_m = np.array(img_m)
                                    img_u = img_u[:, :, np.newaxis]
                                    img_v = img_v[:, :, np.newaxis]
                                    img_m = img_m[:, :, np.newaxis]
                                    OF_img = np.concatenate((img_u, img_v, img_m), axis=2)

                                    exx_img_path = os.path.join(OS_video_path, 'exx.png')
                                    eyy_img_path = os.path.join(OS_video_path, 'eyy.png')
                                    exy_img_path = os.path.join(OS_video_path, 'exy.png')
                                    img_exx = Image.open(exx_img_path)
                                    img_exx = img_exx.resize((224, 224))
                                    img_exx = np.array(img_exx)
                                    img_eyy = Image.open(eyy_img_path)
                                    img_eyy = img_eyy.resize((224, 224))
                                    img_eyy = np.array(img_eyy)
                                    img_exy = Image.open(exy_img_path)
                                    img_exy = img_exy.resize((224, 224))
                                    img_exy = np.array(img_exy)
                                    img_exx = img_exx[:, :, np.newaxis]
                                    img_eyy = img_eyy[:, :, np.newaxis]
                                    img_exy = img_exy[:, :, np.newaxis]
                                    OS_img = np.concatenate((img_exx, img_eyy, img_exy), axis=2)

                                    img = np.concatenate((OF_img, OS_img), axis=2)

                                    selected_channels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
                                    for idx in range(0, len(selected_channels)):
                                        apex_parsing_map_img_path = os.path.join(apex_parsing_map_video_path, 'apex_' + str(selected_channels[idx]) + '_segprob.png')
                                        apex_parsing_map_img = Image.open(apex_parsing_map_img_path)
                                        apex_parsing_map_img = apex_parsing_map_img.resize((224, 224))
                                        apex_parsing_map_img = np.array(apex_parsing_map_img)
                                        apex_parsing_map_img = apex_parsing_map_img[:, :, np.newaxis]

                                        img = np.concatenate((img, apex_parsing_map_img), axis=2)
                                    print(img.shape)

                                    self.test_data.append(img)
                                    self.test_labels.append(test_label)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]

        img_OF = Image.fromarray(img[:, :, :3])
        if self.transform is not None:
            img_OF = self.transform(img_OF)

        img_OS = Image.fromarray(img[:, :, 3:6])
        if self.transform is not None:
            img_OS = self.transform(img_OS)

        img_temp = torch.cat([img_OF, img_OS], 0)
        for i_channel in range(0, 10):
            img_apex_map = Image.fromarray(img[:, :, i_channel+6], 'L')
            if self.transform is not None:
                img_apex_map = self.transform(img_apex_map)
            if self.split == 'Training':
                img_temp = torch.cat([img_temp, img_apex_map], 0)
            if self.split == 'Testing':
                img_temp = torch.cat([img_temp, img_apex_map], 0)
        img = img_temp

        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)
