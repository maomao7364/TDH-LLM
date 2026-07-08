import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import pandas as pd
import torch
from evaluation import *
import matplotlib.pyplot as plt
import matplotlib

def masked_mape_loss(y_pred, y):
    mask = (y) != 0                                         # GCN中y存在小于0的值，原因在于0-1缩放val和test会存在小于0; 不完全处理：scaler拓展到整个data
    return np.abs((y[mask] - y_pred[mask]) / y[mask])

#  调参
train_iter, val_iter, test_iter, mean, std = load_data(1, 1, 1, 1, 64)                                #  args.num_of_hours, args.num_of_days, args.num_of_weeks, 预测间隔, args.batch_size
train_iter_1, val_iter_1, test_iter_1, mean_1, std_1, W, edge_index, edge_weight = data_convert(64)
train_iter_2, val_iter_2, test_iter_2, mean_2, std_2 = load_data(1, 1, 1, 1, 64)
train_iter_3, val_iter_3, test_iter_3, mean_3, std_3 = load_data(1, 0, 0, 1, 64)

file = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-5min\PI-SGTGCN-3.pth'   #  3    3      2(3)      3 (MAPE:2)
file_1 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-5min\STGCN-2.pth'     #  2    3      3(2)      3
file_2 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-5min\ASTGCN-2.pth'    #  2    3      2(1)      3
file_3 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-5min\STGNN-3.pth'     #  3    3      3(3)      3

model = torch.load(file)
model_1 = torch.load(file_1)
model_2 = torch.load(file_2)
model_3 = torch.load(file_3)

model.eval()
model_1.eval()
model_2.eval()
model_3.eval()

y_all = np.empty(shape=(1,num_stations))
y_predict_all = np.empty(shape=(1,num_stations))
y_all_1 = np.empty(shape=(1,num_stations))
y_predict_all_1 = np.empty(shape=(1,num_stations))
y_all_2 = np.empty(shape=(1,num_stations))
y_predict_all_2 = np.empty(shape=(1,num_stations))
y_all_3 = np.empty(shape=(1,num_stations))
y_predict_all_3 = np.empty(shape=(1,num_stations))

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use("seaborn-darkgrid")

with torch.no_grad():                                                   # PI-SGTGCN
    for x, y in test_iter:                                               #    [32, 12, 50, 1]  (32, 50)
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean, std)
        y_all = np.concatenate((y_all, y),axis=0)                        # (33, 50)
        y_pred = model(x)                                                #  (32, 50)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = re_norm(y_pred, mean, std)
        y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)      #  (33, 50)
    y_all = y_all[1:]                                                    # (2005, 50)
    y_predict_all = y_predict_all[1:]

with torch.no_grad():                                                       # STGCN
    for x, y in test_iter_1:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_1, std_1)
        y_all_1 = np.concatenate((y_all_1, y), axis=0)
        y_pred_1 = model_1(x, edge_index, edge_weight).view(len(x), -1)
        y_pred_1 = y_pred_1.cpu().detach().numpy()
        y_pred_1 = re_norm(y_pred_1, mean_1, std_1)
        y_predict_all_1 = np.concatenate((y_predict_all_1, y_pred_1), axis=0)
    y_all_1 = y_all_1[1:]                                                #  (1993, 50)
    y_predict_all_1 = y_predict_all_1[1:]

with torch.no_grad():                                                         # ASTGCN
    for x, y in test_iter_2:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_2, std_2)
        y_all_2 = np.concatenate((y_all_2, y), axis=0)
        y_pred_2 = model_2(x.permute(0, 2, 3, 1))
        y_pred_2 = y_pred_2.cpu().detach().numpy()
        y_pred_2 = re_norm(y_pred_2, mean_2, std_2)
        y_predict_all_2 = np.concatenate((y_predict_all_2, y_pred_2), axis=0)
    y_all_2 = y_all_2[1:]                                                     # (2005, 50)
    y_predict_all_2 = y_predict_all_2[1:]

with torch.no_grad():                                                       # STGNN
    for x, y in test_iter_3:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_3, std_3)
        y_all_3 = np.concatenate((y_all_3, y), axis=0)
        y_pred_3 = model_3(x)
        y_pred_3 = y_pred_3.cpu().detach().numpy()
        y_pred_3 = re_norm(y_pred_3, mean_3, std_3)
        y_predict_all_3 = np.concatenate((y_predict_all_3, y_pred_3), axis=0)
    y_all_3 = y_all_3[1:]                                                 # (2005, 50)
    y_predict_all_3 = y_predict_all_3[1:]

M = np.zeros((50, 3))                                         # PI-SGTGCN
for i in range(50):
    res = np.abs(y_all[:, i] - y_predict_all[:, i])
    mae = np.mean(res)
    mape = np.mean(masked_mape_loss(y_predict_all[:, i], y_all[:, i]))
    rmse = np.sqrt(np.mean(res ** 2))
    M[i][0] = mae
    M[i][1] = mape * 100
    M[i][2] = rmse

M_1 = np.zeros((50, 3))                                        # STGCN
for i in range(50):
    res = np.abs(y_all_1[:, i] - y_predict_all_1[:, i])
    mae = np.mean(res)
    mape = np.mean(masked_mape_loss(y_predict_all_1[:, i], y_all_1[:, i]))
    rmse = np.sqrt(np.mean(res ** 2))
    M_1[i][0] = mae
    M_1[i][1] = mape * 100
    M_1[i][2] = rmse

M_2 = np.zeros((50, 3))                                     # ASTGCN
for i in range(50):
    res = np.abs(y_all_2[:, i] - y_predict_all_2[:, i])
    mae = np.mean(res)
    mape = np.mean(masked_mape_loss(y_predict_all_2[:, i], y_all_2[:, i]))
    rmse = np.sqrt(np.mean(res ** 2))
    M_2[i][0] = mae
    M_2[i][1] = mape * 100
    M_2[i][2] = rmse

M_3 = np.zeros((50, 3))                                      # STGNN
for i in range(50):
    res = np.abs(y_all_3[:, i] - y_predict_all_3[:, i])
    mae = np.mean(res)
    mape = np.mean(masked_mape_loss(y_predict_all_3[:, i], y_all_3[:, i]))
    rmse = np.sqrt(np.mean(res ** 2))
    M_3[i][0] = mae
    M_3[i][1] = mape * 100
    M_3[i][2] = rmse

labels = np.array(range(50))

# MAE, MAPE, RMSE  调参
data = M[:,1]                                           #  (1, 50)
data_1 = M_1[:,1]
data_2 = M_2[:,1]
data_3 = M_3[:,1]
print('PI-SGTGCN_mean: ', data.mean(), 'STGCN_mean: ', data_1.mean(), 'ASTGCN_mean: ', data_2.mean(), 'STGNN_mean: ', data_3.mean())
print('PI-SGTGCN_min: ', np.argmin(data), 'PI-SGTGCN_max: ', np.argmax(data))

num = 0
num_1 = 0
num_2 = 0
num_3 = 0
for i in data:
    if i < 11:
        num += 1
for i in data_1:
    if i < 11:
        num_1 += 1
for i in data_2:
    if i < 11:
        num_2 += 1
for i in data_3:
    if i < 11:
        num_3 += 1
print('PI-SGTGCN: ', num, 'STGCN: ', num_1, 'ASTGCN: ', num_2, 'STGNN: ', num_3)

angles = np.linspace(0,2*np.pi, 50, endpoint=False)    #   根据数据长度平均分割圆周长
angles = np.concatenate((angles,[angles[0]]))
labels=np.concatenate((labels,[labels[0]]))              #对labels进行封闭

data = np.concatenate((data,[data[0]]))
data_1 = np.concatenate((data_1,[data[0]]))
data_2 = np.concatenate((data_2,[data[0]]))
data_3 = np.concatenate((data_3,[data[0]]))

fig = plt.figure(facecolor="white")                   #facecolor 设置框体的颜色
ax = plt.subplot(111,polar=True)                            #将图分成1行1列，画出位置1的图；设置图形为极坐标图

ax.plot(angles, data, linestyle='-', color ='#881C96', linewidth=3, label='PI-SGTGCN')    # PI-SGTGCN
ax.plot(angles, data_1, linestyle='-', color ='#428AC9', linewidth=3, alpha=0.8, label='STGCN')      # STGCN
ax.plot(angles, data_2, linestyle='-', color ='#F37E78', linewidth=3, alpha=0.8, label='ASTGCN')      #  ASTGCN
ax.plot(angles, data_3, linestyle='-', color ='#129392', linewidth=3, alpha=0.8, label='STGNN')       #  STGNN

plt.fill(angles,data,facecolor='#881C96',alpha=0.3)    #填充两条线之间的色彩，alpha为透明度
plt.fill(angles,data_1,facecolor='#428AC9',alpha=0.1)
plt.fill(angles,data_2,facecolor='#F37E78',alpha=0.1)
plt.fill(angles,data_3,facecolor='#129392',alpha=0.1)

plt.legend(loc=[0.95, 0.95], fontsize='large')

ax.set_yticks(np.linspace(4, 16, 4))
plt.thetagrids(angles*180/np.pi,labels)                 #做标签
plt.show()



