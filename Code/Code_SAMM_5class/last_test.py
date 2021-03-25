from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import transforms as transforms
import numpy as np
import os
from dataload import Dataload
from torch.autograd import Variable
from models import KFC_MER_Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall


os.environ['CUDA_VISIBLE_DEVICES'] = "2"
use_cuda = True
torch.backends.cudnn.benchmark = True


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def confusion_matrix_excel(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    data = pd.DataFrame(cm)
    writer = pd.ExcelWriter('confusion_matrix.xlsx')
    data.to_excel(writer, 'cm', float_format='%.5f')
    writer.save()
    writer.close()
    return cm


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='15')

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    print(cm.shape[1])
    print(cm.shape[0])
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), title=title, ylabel='True label', xlabel='Predicted label')

    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticklabels(['anger', 'contempt', 'happiness', 'surprise', 'other'], fontsize='small')
    ax.set_yticklabels(['anger', 'contempt', 'happiness', 'surprise', 'other'], fontsize='small')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'g'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(round(cm[i, j], 2), fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg')
    plt.show()


model_dir = '../../Model_parameters/SAMM_5class/net/'
all_sub_list = ['006', '007', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                '021', '022', '023', '025', '026', '028', '030', '031', '032', '033', '034', '035', '037']


correct = 0
total = 0
y_true = []
y_pre = []
for leave_out_sub in all_sub_list:
    print('==> Leaving out ' + leave_out_sub)

    net = KFC_MER_Model()
    model_path = os.path.join(model_dir, leave_out_sub, 'Resnest18_2blocks_OFOS_map_49.pth')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    if use_cuda:
        net.cuda()

    test_set = Dataload(split='Testing', transform=transform_test, leave_out=leave_out_sub)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=1)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs1 = inputs[:, :3, :, :]
        inputs2 = inputs[:, 3:6, :, :]
        seg_map = inputs[:, 6:, :, :]

        if use_cuda:
            inputs1, inputs2, seg_map, targets = inputs1.cuda(), inputs2.cuda(), seg_map.cuda(), targets.cuda()
        with torch.no_grad():
            inputs1, inputs2, seg_map, targets = Variable(inputs1), Variable(inputs2), Variable(seg_map), Variable(targets)

        outputs = net(inputs1, seg_map, inputs2, seg_map)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print(predicted.eq(targets.data).cpu().sum())

        y_true.append(targets.cpu().numpy())
        y_pre.append(predicted.cpu().numpy())

# calculate accuracy
print('total:', total, 'correct:', correct)
Test_acc = 100 * float(correct) / total
print('Test accuracy: %0.6f' % Test_acc)

# confusion matrix
y_true = np.concatenate(y_true, axis=0)
y_pre = np.concatenate(y_pre, axis=0)
cm = confusion_matrix_excel(y_true=y_true, y_pred=y_pre)
print(cm)

##############################
f1_score, precision, recall = fpr(cm, n_exp=5)
print('f1_score:', f1_score, 'precision:', precision, 'recall:', recall)
WAR = weighted_average_recall(cm, n_exp=5, class_num_list=[57, 12, 26, 15, 26])
UAR = unweighted_average_recall(cm, n_exp=5)

plot_Matrix(cm, classes=5, title=None, cmap=plt.cm.Blues)



