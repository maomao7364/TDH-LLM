import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluation import *
from data_preprocess import data_convert


###################实验一#############################
# 语言
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.style.use("seaborn-darkgrid")
# fig = plt.figure(figsize=(8,6))
# colors1 = ['#D1E6F8', '#A1CAED', '#6FAEE0', '#224767']
# colors2 = ['#E4CCE4', '#CA95C3', '#A85EA2', '#8E338A']
# font = {'family': 'Times New Roman'}

# 子图1
# aver1 = [21.07, 9.70, 31.80]
# aver2 = [21.27, 10.3, 31.99]
# aver3 = [20.87, 9.67, 31.58]
# aver4 = [20.45, 9.33, 31.2]
#
# x_label = ['MAE', 'MAPE(%)', 'RMSE']
# total_width, n = 0.8, 5
# # 每种类型的柱状图宽度
# width = total_width / n
# x_col = np.arange(3)
#
# # 重新设置x轴的坐标
# x_col = x_col - (total_width - width) / 2
#
# p1 = fig.add_subplot(221)
#
# 画柱状图
# plt.bar(x_col, aver1, width=width, color=colors2[0], label='SGTGCN')
# plt.bar(x_col + width, aver2, color=colors2[1], width=width, label='PI-SGTGCN-noGRU')
# plt.bar(x_col + 2*width, aver3, color=colors2[2], width=width, label='PI-SGTGCN-noTrans')
# plt.bar(x_col + 3*width, aver4, color=colors2[3], width=width, label='PI-SGTGCN')
#
# plt.xticks(np.arange(3), size=15, weight='bold')
# plt.xticks(np.arange(3), x_label)
# plt.yticks(size=15, weight='bold')
#
# 显示图例
# p1.legend(loc='best', fontsize='large', frameon=True, edgecolor='black', prop=font)

# train_iter, val_iter, test_iter, mean, std = load_data(1, 1, 1, 12, 64)                                #  args.num_of_hours, args.num_of_days, args.num_of_weeks, 预测间隔, args.batch_size
# file = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\PI-SGTGCN-3.pth'
# file_1 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\PI-SGTGCN-noGRU-3.pth'
# file_2 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\SGTGCN-2.pth'
# file_3 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\PI-SGTGCN-noTrans-2.pth'
# model = torch.load(file)
# model_1 = torch.load(file_1)
# model_2 = torch.load(file_2)
# model_3 = torch.load(file_3)
#
# y_all = np.empty(shape=(1,num_stations))
# y_predict_all = np.empty(shape=(1,num_stations))
# y_predict_all_1 = np.empty(shape=(1,num_stations))
# y_predict_all_2 = np.empty(shape=(1,num_stations))
# y_predict_all_3 = np.empty(shape=(1,num_stations))
#
# model.eval()
# model_1.eval()
# model_2.eval()
# model_3.eval()
#
# with torch.no_grad():
#     mae, mape, mse = [], [], []
#     for x, y in test_iter:                                            #    [32, 12, 50, 1]  (32, 50)
#         y = y.cpu().detach().numpy()
#         y = re_norm(y, mean, std)
#         y_all = np.concatenate((y_all, y),axis=0)                      # (33, 50)
#
#         y_pred = model(x)                                                #  (32, 50)
#         y_pred_1 = model_1(x)
#         y_pred_2 = model_2(x)
#         y_pred_3 = model_3(x)
#
#         y_pred = y_pred.cpu().detach().numpy()
#         y_pred = re_norm(y_pred, mean, std)
#         y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)    #  (33, 50)
#
#         y_pred_1 = y_pred_1.cpu().detach().numpy()
#         y_pred_1 = re_norm(y_pred_1, mean, std)
#         y_predict_all_1 = np.concatenate((y_predict_all_1, y_pred_1), axis=0)  # (33, 50)
#
#         y_pred_2 = y_pred_2.cpu().detach().numpy()
#         y_pred_2 = re_norm(y_pred_2, mean, std)
#         y_predict_all_2 = np.concatenate((y_predict_all_2, y_pred_2), axis=0)  # (33, 50)
#
#         y_pred_3 = y_pred_3.cpu().detach().numpy()
#         y_pred_3 = re_norm(y_pred_3, mean, std)
#         y_predict_all_3 = np.concatenate((y_predict_all_3, y_pred_3), axis=0)  # (33, 50)
#
#         d = np.abs(y - y_pred)
#         mae += d.tolist()
#         mape += masked_mape_loss(y_pred, y).tolist()
#         mse += (d ** 2).tolist()
#
#     # 去掉都是0的第一行
#     y_all = y_all[1:]                      # (2005, 50)
#     y_predict_all = y_predict_all[1:]     # (2005, 50)
#     y_predict_all_1 = y_predict_all_1[1:]  # (2005, 50)
#     y_predict_all_2 = y_predict_all_2[1:]  # (2005, 50)
#     y_predict_all_3 = y_predict_all_3[1:]  # (2005, 50)
#
# # 计算MAE的极小值
# M = np.zeros((50, 3))
# for i in range(50):
#     res = np.abs(y_all[:, i]-y_predict_all[:, i])
#     Max = np.max(res)
#     Min = np.min(res)
#     Mae = np.mean(res)
#     M[i][0] = Max
#     M[i][1] = Min
#     M[i][2] = Mae
#
# M_1 = np.zeros((50, 3))
# for i in range(50):
#     res_1 = np.abs(y_all[:, i]-y_predict_all_1[:, i])
#     Max_1 = np.max(res_1)
#     Min_1 = np.min(res_1)
#     Mae_1 = np.mean(res_1)
#     M_1[i][0] = Max_1
#     M_1[i][1] = Min_1
#     M_1[i][2] = Mae_1
#
# M_2 = np.zeros((50, 3))
# for i in range(50):
#     res_2 = np.abs(y_all[:, i]-y_predict_all_2[:, i])
#     Max_2 = np.max(res_2)
#     Min_2 = np.min(res_2)
#     Mae_2 = np.mean(res_2)
#     M_2[i][0] = Max_2
#     M_2[i][1] = Min_2
#     M_2[i][2] = Mae_2
#
# M_3 = np.zeros((50, 3))
# for i in range(50):
#     res_3 = np.abs(y_all[:, i]-y_predict_all_3[:, i])
#     Max_3 = np.max(res_3)
#     Min_3 = np.min(res_3)
#     Mae_3 = np.mean(res_3)
#     M_3[i][0] = Max_3
#     M_3[i][1] = Min_3
#     M_3[i][2] = Mae_3
#
# p1 = fig.add_subplot(221)
#
# plt.plot(M_2[:, 2], marker='o', linestyle="--", color=colors1[0], linewidth=2, markersize=7, label='SGTGCN')
# plt.plot(M_1[:, 2], marker='*', linestyle="--", color=colors1[1], linewidth=2, markersize=7, label='PI-SGTGCN-noGRU')
# plt.plot(M_3[:, 2], marker='v', linestyle="--", color=colors1[2], linewidth=2, markersize=7, label='PI-SGTGCN-noTrans')
# plt.plot(M[:, 2], marker='s', linestyle="--", color=colors1[3], linewidth=2, markersize=7, label='PI-SGTGCN')
#
# plt.xlabel('Stations',fontsize=18,fontweight='bold')
# plt.ylabel('MAE',fontsize=18,fontweight='bold')
# plt.xticks(size=15, weight='bold')
# plt.yticks(size=15, weight='bold')
# p1.legend(loc='best', fontsize=13, frameon=True, edgecolor='blue', prop=font)

# 残差sns图
# Res = y_all - y_predict_all
# Res_1 = y_all - y_predict_all_1
# Res_2 = y_all - y_predict_all_2
# Res_3 = y_all - y_predict_all_3
# x_data = np.array([range(50)] * 2005).T.flatten()                 #  (50*2005)
# y_data = np.array(Res).T.flatten()                                #  (50*2005)
# y_data_1 = np.array(Res_1).T.flatten()
# y_data_2 = np.array(Res_2).T.flatten()
# y_data_3 = np.array(Res_3).T.flatten()
# data_all = pd.DataFrame({'Station': x_data, 'all': y_data, 'noGRU': y_data_1, 'noPI': y_data_2, 'noTrans': y_data_3})
# data = pd.DataFrame({'Station': x_data, 'Res': y_data})
# data_1 = pd.DataFrame({'Station': x_data, 'Res_1': y_data_1})
# data_2 = pd.DataFrame({'Station': x_data, 'Res_2': y_data_2})
# data_3 = pd.DataFrame({'Station': x_data, 'Res_3': y_data_3})

# 均值和95%置信区间
# sns.relplot(x="Station", y="value", hue='variable', kind="line", data=data_all.melt(id_vars='Station'),  errorbar="ci")
# sns.relplot(data=data_1, kind="line", x="Station", y="Res_1", errorbar="sd")

# 均值和标准差
# sns.relplot(x="Station", y="value", hue='variable', kind="line", data=data_all.melt(id_vars='Station'),  errorbar="sd")
# sns.relplot(data=data_1, kind="line", x="Station", y="Res_1", errorbar="sd")

# plt.legend(['all均值', "all的95%置信区间", 'noGRU均值', "noGRU的95%置信区间", 'noPI均值', "noPI的95%置信区间", 'noTrans均值', "noTrans的95%置信区间",], loc='best')
plt.show()


# 子图2,3,4
# x = np.array([5, 15, 30, 60])
#
# # 子图2
# mae1 = [18.99, 20.55, 21.44, 23.29]
# mae2 = [18.66, 20.71, 21.75, 23.97]
# mae3 = [18.61, 20.40, 21.29, 23.18]
# mae4 = [18.06, 20.01, 20.95, 22.76]
#
# p2 = fig.add_subplot(222)
# p2.plot(x,mae1, marker='o', linestyle="--",color=colors1[0],linewidth=2.0, markersize=7, label='SGTGCN')
# p2.plot(x,mae2, marker='*', linestyle="--", color=colors1[1], linewidth=2.0, markersize=7, label='PI-SGTGCN-noGRU')
# p2.plot(x,mae3, marker='v', linestyle="--", color=colors1[2], linewidth=2.0, markersize=7, label='PI-SGTGCN-noTrans')
# p2.plot(x,mae4, marker='s', linestyle="--", color=colors1[3], linewidth=2.0, markersize=7, label='PI-SGTGCN')
#
# # 给坐标轴加上标签
# plt.xlabel('Prediction Interval(mins)',fontsize=18,fontweight='bold')
# plt.ylabel('MAE',fontsize=18,fontweight='bold')
# plt.xticks(x, size=15, weight='bold')
# plt.yticks(size=15, weight='bold')

# 标题
# p2.legend(loc='best', fontsize='large', frameon=True, edgecolor='blue', prop=font)
#
# # 子图3
# mape1 = [8.89, 9.22, 9.73, 10.95]
# mape2 = [8.94, 10.01, 10.24, 12.00]
# mape3 = [8.70, 9.45, 9.83, 10.69]
# mape4 = [8.13, 9.01, 9.68, 10.49]
#
# p3 = fig.add_subplot(223)
# p3.plot(x,mape1, marker='o', linestyle="--",color=colors1[0],linewidth=2.0, markersize=7, label='SGTGCN')
# p3.plot(x,mape2, marker='*', linestyle="--", color=colors1[1], linewidth=2.0, markersize=7, label='PI-SGTGCN-noGRU')
# p3.plot(x,mape3, marker='v', linestyle="--", color=colors1[2], linewidth=2.0, markersize=7, label='PI-SGTGCN-noTrans')
# p3.plot(x,mape4, marker='s', linestyle="--", color=colors1[3], linewidth=2.0, markersize=7, label='PI-SGTGCN')
#
# # 给坐标轴加上标签
# plt.xlabel('Prediction Interval(mins)',fontsize=18,fontweight='bold')
# plt.ylabel('MAPE(%)',fontsize=18,fontweight='bold')
# plt.xticks(x, size=15, weight='bold')
# plt.yticks(size=15, weight='bold')
#
# # 标题
# p3.legend(loc='best', fontsize='large', frameon=True, edgecolor='blue', prop=font)
#
# # 子图4
# rmse1 = [28.47, 31.26, 32.51, 34.94]
# rmse2 = [28.10, 31.23, 32.91, 35.72]
# rmse3 = [27.98, 30.95, 32.45, 34.93]
# rmse4 = [27.58, 30.58, 32.15, 34.49]
#
# p4 = fig.add_subplot(224)
# p4.plot(x,rmse1, marker='o', linestyle="--",color=colors1[0],linewidth=2.0, markersize=7, label='SGTGCN')
# p4.plot(x,rmse2, marker='*', linestyle="--", color=colors1[1], linewidth=2.0, markersize=7, label='PI-SGTGCN-noGRU')
# p4.plot(x,rmse3, marker='v', linestyle="--", color=colors1[2], linewidth=2.0, markersize=7, label='PI-SGTGCN-noTrans')
# p4.plot(x,rmse4, marker='s', linestyle="--", color=colors1[3], linewidth=2.0, markersize=7, label='PI-SGTGCN')
#
# # 给坐标轴加上标签
# plt.xlabel('Prediction Interval(mins)',fontsize=18,fontweight='bold')
# plt.ylabel('RMSE',fontsize=18,fontweight='bold')
# plt.xticks(x, size=15, weight='bold')
# plt.yticks(size=15, weight='bold')
#
# # 标题
# p4.legend(loc='best', fontsize='large', frameon=True, edgecolor='blue', prop=font)
#
# plt.subplots_adjust(wspace=0.1, hspace=0.3)
# plt.show()



###################实验三#############################
# def masked_mape_loss(y_pred, y):
#     mask = (y) != 0                                         # GCN中y存在小于0的值，原因在于0-1缩放val和test会存在小于0; 不完全处理：scaler拓展到整个data
#     return np.abs((y[mask] - y_pred[mask]) / y[mask])
#
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['font.size'] = 15
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# fig = plt.figure(figsize=(15,12))
# font = {'family': 'Times New Roman'}
#
# train_iter, val_iter, test_iter, mean, std = load_data(1, 1, 1, 12, 64)                                #  args.num_of_hours, args.num_of_days, args.num_of_weeks, 预测间隔, args.batch_size
# speed = pd.read_csv(r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\717093_test_speed.csv')
# speed = np.array(speed).flatten()                         #  (2005)
#
# x_date_60 = pd.date_range('202106240055', periods = 2005, freq = '5MIN').strftime("%D %H:%M").tolist()    #  60分钟预测
# id_station = 9   # VDS717093
# file = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\PI-SGTGCN-3.pth'
# model = torch.load(file)
#
# y_all = np.empty(shape=(1,num_stations))
# y_predict_all = np.empty(shape=(1,num_stations))
# model.eval()
# with torch.no_grad():
#     mae, mape, mse = [], [], []
#     for x, y in test_iter:                                            #    [32, 12, 50, 1]  (32, 50)
#         y = y.cpu().detach().numpy()
#         y = re_norm(y, mean, std)
#         y_all = np.concatenate((y_all, y),axis=0)                      # (33, 50)
#
#         y_pred = model(x)                                                #  (32, 50)
#
#         y_pred = y_pred.cpu().detach().numpy()
#         y_pred = re_norm(y_pred, mean, std)
#         y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)    #  (33, 50)
#
#         d = np.abs(y - y_pred)
#         mae += d.tolist()
#         mape += masked_mape_loss(y_pred, y).tolist()
#         mse += (d ** 2).tolist()
#
#     MAE = np.array(mae).mean()
#     MAPE = np.mean(mape)
#     RMSE = np.sqrt(np.array(mse).mean())
#
#     # 去掉都是0的第一行
#     y_all = y_all[1:]                      # (2005, 50)
#     y_predict_all = y_predict_all[1:]     # (2005, 50)
#
#     y_all_mean = np.mean(y_all, axis=1) # 对各行求均值 （2005）
#     y_predict_all_mean = np.mean(y_predict_all, axis=1) # 对各行求均值 （2005）
#
#     p1 = fig.add_subplot(211)
#     p1.plot(x_date_60, y_all_mean, color='#6A1128', linestyle="-", linewidth=2.5)
#     p1.plot(x_date_60, y_predict_all_mean, color='#428AC9', linestyle="--", linewidth=2.5)
#     p1.legend(["Ground Truth", 'Prediction'], loc='lower right', fontsize=15, edgecolor='#B4B7B7')
#     plt.xticks(range(0, len(x_date_60), 288), size=15, weight='bold')
#     plt.yticks(size=15, weight='bold')
#     plt.xlabel('Times', size=18, weight='bold')
#     plt.ylabel('Traffic flow', size=18, weight='bold')
#     plt.title('Averages of all stations', fontsize=18, fontweight='bold')
#     plt.grid()
#
#     p2 = fig.add_subplot(212)
#     p2_2 = p2.twinx()                      # 创建第二个坐标轴
#
#     p2.plot(x_date_60, y_all[:, id_station], color='#27744B', linestyle="-", linewidth=2.5, label='Ground Truth')
#     p2.plot(x_date_60, y_predict_all[:, id_station], color='#A3BFAD', linestyle="--", linewidth=2.5, label='Prediction')
#     p2_2.plot(x_date_60, speed, color='#F37E78', linestyle="-", linewidth=2.5, label='Speed')
#
#     p2.legend(loc='best', fontsize=15, edgecolor='#B4B7B7')
#     p2_2.legend(loc='lower right', fontsize=15, edgecolor='#B4B7B7')
#
#     p2.set_xlabel('Times', size=18, weight='bold')
#     p2.set_ylabel('Traffic flow', size=18, weight='bold')
#     p2_2.set_ylabel('Traffic speed', size=18, weight='bold')
#
#     plt.xticks(range(0, len(x_date_60), 288), size=15, weight='bold')
#     plt.yticks(size=15, weight='bold')
#     plt.title('VDS 717093', fontsize=18, fontweight='bold')
#     plt.grid()
#
# plt.subplots_adjust(hspace=0.25)
# plt.show()


###################实验四#############################
def masked_mape_loss(y_pred, y):
    mask = (y) != 0                                         # GCN中y存在小于0的值，原因在于0-1缩放val和test会存在小于0; 不完全处理：scaler拓展到整个data
    return np.abs((y[mask] - y_pred[mask]) / y[mask])

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure(figsize=(15,12))
font = {'family': 'Times New Roman'}

train_iter, val_iter, test_iter, mean, std = load_data(1, 1, 1, 12, 64)                                #  args.num_of_hours, args.num_of_days, args.num_of_weeks, 预测间隔, args.batch_size
train_iter_1, val_iter_1, test_iter_1, mean_1, std_1, W, edge_index, edge_weight = data_convert(64)
train_iter_2, val_iter_2, test_iter_2, mean_2, std_2 = load_data(1, 1, 1, 12, 64)
train_iter_3, val_iter_3, test_iter_3, mean_3, std_3 = load_data(1, 0, 0, 12, 64)

speed = pd.read_csv(r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\717093_test_speed.csv')
speed = np.array(speed).flatten()                         #  (2005)

file = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\PI-SGTGCN-3.pth'   # 3
file_1 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\STGCN-3.pth'     # 3
file_2 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\ASTGCN-3.pth'    # 3
file_3 = r'C:\Users\Administrator\Desktop\GCN_Reinforcement\DDPG+GCN\small\result-60min\STGNN-3.pth'     # 3

model = torch.load(file)
model_1 = torch.load(file_1)
model_2 = torch.load(file_2)
model_3 = torch.load(file_3)

y_all = np.empty(shape=(1,num_stations))
y_predict_all = np.empty(shape=(1,num_stations))
y_all_1 = np.empty(shape=(1,num_stations))
y_predict_all_1 = np.empty(shape=(1,num_stations))
y_all_2 = np.empty(shape=(1,num_stations))
y_predict_all_2 = np.empty(shape=(1,num_stations))
y_all_3 = np.empty(shape=(1,num_stations))
y_predict_all_3 = np.empty(shape=(1,num_stations))

model.eval()
model_1.eval()
model_2.eval()
model_3.eval()

with torch.no_grad():                                      # PI-SGTGCN
    for x, y in test_iter:                                            #    [32, 12, 50, 1]  (32, 50)
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean, std)
        y_all = np.concatenate((y_all, y),axis=0)                      # (33, 50)
        y_pred = model(x)                                                #  (32, 50)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = re_norm(y_pred, mean, std)
        y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)    #  (33, 50)
    y_all = y_all[1:]                                                  # (2005, 50)
    y_predict_all = y_predict_all[1:]

with torch.no_grad():                                       # STGCN
    for x, y in test_iter_1:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_1, std_1)
        y_all_1 = np.concatenate((y_all_1, y), axis=0)
        y_pred_1 = model_1(x, edge_index, edge_weight).view(len(x), -1)
        y_pred_1 = y_pred_1.cpu().detach().numpy()
        y_pred_1 = re_norm(y_pred_1, mean_1, std_1)
        y_predict_all_1 = np.concatenate((y_predict_all_1, y_pred_1), axis=0)
    y_all_1 = y_all_1[1:]                                             #    (1993, 50)
    y_predict_all_1 = y_predict_all_1[1:]                            #    (1993, 50)

with torch.no_grad():                                       # ASTGCN
    for x, y in test_iter_2:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_2, std_2)
        y_all_2 = np.concatenate((y_all_2, y), axis=0)
        y_pred_2 = model_2(x.permute(0, 2, 3, 1))
        y_pred_2 = y_pred_2.cpu().detach().numpy()
        y_pred_2 = re_norm(y_pred_2, mean_2, std_2)
        y_predict_all_2 = np.concatenate((y_predict_all_2, y_pred_2), axis=0)
    y_all_2 = y_all_2[1:]                                           # (2005, 50)
    y_predict_all_2 = y_predict_all_2[1:]                           # (2005, 50)

with torch.no_grad():                                     # STGNN
    for x, y in test_iter_3:
        y = y.cpu().detach().numpy()
        y = re_norm(y, mean_3, std_3)
        y_all_3 = np.concatenate((y_all_3, y), axis=0)
        y_pred_3 = model_3(x)
        y_pred_3 = y_pred_3.cpu().detach().numpy()
        y_pred_3 = re_norm(y_pred_3, mean_3, std_3)
        y_predict_all_3 = np.concatenate((y_predict_all_3, y_pred_3), axis=0)
    y_all_3 = y_all_3[1:]                                    # (2005, 50)
    y_predict_all_3 = y_predict_all_3[1:]                   # (2005, 50)

    # 60分钟预测
    # plt.style.use("seaborn-darkgrid")
    # i = 276   # 276 (25日0点)  1160 (28日0点)
    # m = 0     # 0
    # n = 874   # 288*3+10=874  288*2=576
    #
    # id_station = 5       #  best: 5   worse: 18
    # x_date_60 = pd.date_range('202106250000', periods=n, freq='5MIN').strftime("%m-%d %H:%M").tolist()    # 60分钟预测
    #
    # plt.scatter(x_date_60, y_all[i+m:i+m+n, id_station], color='gray', linewidth=2.5, label='dot')                          #  276 (25日0点)  1160 (28日0点)
    # plt.plot(x_date_60, y_all[i + m:i + m + n, id_station], color='red', linewidth=2.5, label='line')                 # 276 (25日0点)  1160 (28日0点)
    #
    # # plt.plot(x_date_60, y_predict_all[i+m:i+m+n, id_station], color='#881C96', linestyle="-", linewidth=2.5, label='PI-SGTGCN')
    # # plt.plot(x_date_60, y_predict_all_1[i-12+m:i-12+m+n, id_station], color='#428AC9', linestyle="-", linewidth=2.5, label='STGCN', alpha=0.7)    #  265 (25日0点)  选择26,27,28三天展示，加上10是突显日期
    # # plt.plot(x_date_60, y_predict_all_2[i+m:i+m+n, id_station], color='#F37E78', linestyle="-", linewidth=2.5, label='ASTGCN', alpha=0.7)    # 小窗口展示 6:00-12:00
    # # plt.plot(x_date_60, y_predict_all_3[i+m:i+m+n, id_station], color='#129392', linestyle="-", linewidth=2.5, label='STGNN', alpha=0.7)
    #
    # plt.xlabel('Times', size=16, weight='bold')
    # plt.ylabel('Traffic flow(veh/5min)', size=16, weight='bold')
    # plt.xticks(range(0, len(x_date_60), 144), size=15, weight='bold')
    # plt.yticks(size=15, weight='bold')
    # plt.title('Station 18', fontsize=16, fontweight='bold')
    # plt.legend(loc='best', fontsize=15, frameon=True, edgecolor='black')
    # plt.show()


    # 60分钟预测 小窗口
    plt.style.use("seaborn-darkgrid")

    m = 696  #  72        696   |  72   588
    n = 65  #   60+5=65   65    |  65    65

    id_station = 5         #  best: 5   worse: 18
    x_date_60 = pd.date_range('202106271000', periods=n, freq='5MIN').strftime("%H:%M").tolist()            # 06250600  06271000  06250600  06270100

    plt.plot(x_date_60, y_all[276+m:276+m+n, id_station], color='gray', linestyle="--", linewidth=2.5, label='Ground Truth')
    plt.plot(x_date_60, y_predict_all[276+m:276+m+n, id_station], color='#881C96', linestyle="-", linewidth=2.5, label='PI-SGTGCN')
    plt.plot(x_date_60, y_predict_all_1[264+m:264+m+n, id_station], color='#428AC9', linestyle="-", linewidth=2.5, label='STGCN', alpha=0.7)
    plt.plot(x_date_60, y_predict_all_2[276+m:276+m+n, id_station], color='#F37E78', linestyle="-", linewidth=2.5, label='ASTGCN', alpha=0.7)
    plt.plot(x_date_60, y_predict_all_3[276+m:276+m+n, id_station], color='#129392', linestyle="-", linewidth=2.5, label='STGNN', alpha=0.7)

    plt.xlabel('Times', size=15, weight='bold')
    plt.ylabel('Traffic flow(veh/5min)', size=15, weight='bold')
    plt.xticks(range(0, len(x_date_60), 12), size=14, weight='bold')
    plt.yticks(size=14, weight='bold')
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    plt.show()

    # 计算模型的残差和
    res = np.sum(np.abs(y_all[276 + m:276 + m + 60, id_station] - y_predict_all[276 + m:276 + m + 60, id_station]))
    res_1 = np.sum(np.abs(y_all[276 + m:276 + m + 60, id_station] - y_predict_all_1[264+m:264 + m + 60, id_station]))
    res_2 = np.sum(np.abs(y_all[276 + m:276 + m + 60, id_station] - y_predict_all_2[276+m:276 + m + 60, id_station]))
    res_3 = np.sum(np.abs(y_all[276 + m:276 + m + 60, id_station] - y_predict_all_3[276+m:276 + m + 60, id_station]))
    print('PI-SGTGCN的残差和：', res)
    print('STGCN的残差和：', res_1)
    print('ASTGCN的残差和：', res_2)
    print('STGNN的残差和：', res_3)

#
    # 30分钟预测 小窗口
    # m = 708  # 60， 708
    # n = 50  # 50， 50
    #
    # id_station = 10    # VDS717240:11  VDS777066: 50
    # x_date_30 = pd.date_range('202106251100', periods=n, freq='5MIN').strftime("%H:%M").tolist()  # 30分钟预测  06-25 12:00  06-27 12:00
    #
    # plt.scatter(x_date_30, y_all[282 + m:282 + m + n, id_station], color='gray', linewidth=2.5, label='Ground Truth')  # 282 (25日0点)  选择26,27,28三天展示，加上10是突显日期
    # plt.plot(x_date_30, y_predict_all[282 + m:282 + m + n, id_station], color='#881C96', linestyle="-", linewidth=2.5, label='PI-SGTGCN')
    # plt.plot(x_date_30, y_predict_all_1[270+m:270+m+n, id_station], color='#FFCC4F', linestyle="-", linewidth=2.5, label='STGCN', alpha=0.5)
    # plt.plot(x_date_30, y_predict_all_2[282 + m:282 + m + n, id_station], color='#F37E78', linestyle="-", linewidth=2.5, label='ASTGCN', alpha=0.5)  # 小窗口展示 6:00-24:00
    # plt.plot(x_date_30, y_predict_all_3[282+m:282+m+n, id_station], color='#27744B', linestyle="-", linewidth=2.5, label='STGNN', alpha=0.5)
    #
    # plt.xticks(range(0, len(x_date_30), 12), size=13, weight='bold')
    # plt.yticks(size=13, weight='bold')
    # plt.show()

    # 30分钟预测
    # plt.style.use("seaborn-darkgrid")
    # m = 0  # 0,
    # n = 874  # 874
    # id_station = 10    # VDS717240:11
    # x_date_30 = pd.date_range('202106250000', periods=n, freq='5MIN').strftime("%m-%d %H:%M").tolist()  # 30分钟预测
    #
    # plt.scatter(x_date_30, y_all[282 + m:282 + m + n, id_station], color='gray', linewidth=2.5, label='Ground Truth')  # 282 (25日0点)  选择26,27,28三天展示，加上10是突显日期
    # plt.plot(x_date_30, y_predict_all[282 + m:282 + m + n, id_station], color='#881C96', linestyle="-", linewidth=2.5, label='PI-SGTGCN')
    # plt.plot(x_date_30, y_predict_all_1[270+m:270+m+n, id_station], color='#FFCC4F', linestyle="-", linewidth=2.5, label='STGCN', alpha=0.5)
    # plt.plot(x_date_30, y_predict_all_2[282 + m:282 + m + n, id_station], color='#F37E78', linestyle="-", linewidth=2.5, label='ASTGCN', alpha=0.5)  # 小窗口展示 6:00-24:00
    # plt.plot(x_date_30, y_predict_all_3[282+m:282+m+n, id_station], color='#27744B', linestyle="-", linewidth=2.5, label='STGNN', alpha=0.5)
    #
    # plt.xlabel('Times', size=19, weight='bold')
    # plt.ylabel('Traffic flow(veh/5min)', size=19, weight='bold')
    # plt.xticks(range(0, len(x_date_30), 144), size=17, weight='bold')
    # plt.yticks(size=17, weight='bold')
    # plt.title('VDS 717240', fontsize=19, fontweight='bold')
    # plt.legend(loc='best', fontsize=15, frameon=True, edgecolor='black')
    # plt.show()



















# p2 = plt.twinx()                      # 创建第二个坐标轴
    # p2.plot(x_date_60, speed, color='#F37E78', linestyle="-", linewidth=2.5, label='Speed')
    # p2.set_ylabel('Traffic speed', size=18, weight='bold')
    #
    # # 合并两个坐标轴的图例
    # lines, labels = plt.gca().get_legend_handles_labels()
    # lines2, labels2 = p2.get_legend_handles_labels()
    # p2.legend(lines + lines2, labels + labels2, loc='best', fontsize=15, edgecolor='#B4B7B7')












# y_all_mean = np.mean(y_all, axis=1) # 对各行求均值 （2005）
# y_predict_all_mean = np.mean(y_predict_all, axis=1) # 对各行求均值 （2005）




# # 设置x轴的刻度
# plt.xlim(0,20)
# plt.ylim(0,400)


































# R = torch.relu(torch.softmax(a, -1)) + torch.eye(x.shape[1]).to(device)  # relu稀疏化降低计算复杂度  a +  I = A
#             D = (R.sum(-1) ** -0.5)
#             D[torch.isinf(D)] = 0.
#             D = torch.diag_embed(D)  # 变对角矩阵
#             A = torch.matmul(torch.matmul(D, R), D)       #   D_(-0.5) * A * D_(-0.5)
#             x = torch.relu(self.ff1(torch.matmul(A, x)))  # [16, 307, 307] * [16, 307, 64] = [16, 307, 64]


# def evaluate_metric_gcn(model, W, data_iter, scaler, id_station):    #  [559, 50]  [559, 12, 50, 1]
#     y_all = np.empty(shape=(1,num_stations))
#     y_predict_all = np.empty(shape=(1,num_stations))
#     model.eval()
#     with torch.no_grad():
#         mae, mape, mse = [], [], []
#         for x, y in data_iter:                                                   #    [32, 12, 50, 1]  (32, 50)
#             y = scaler.inverse_transform(y.cpu().numpy())                        # (32, 50)
#             y_all = np.concatenate((y_all, y),axis=0)                            # (33, 50)
#             y_pred = scaler.inverse_transform(model(x, W).cpu().numpy())         #  (32, 50)
#
#             y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)       #  (33, 50)
#
#             d = np.abs(y - y_pred)
#             mae += d.tolist()
#             mape += (d / y).tolist()
#             mse += (d ** 2).tolist()
#
#         MAE = np.array(mae).mean()
#         MAPE = np.array(mape).mean()
#         RMSE = np.sqrt(np.array(mse).mean())
#
#         # 去掉都是0的第一行
#         y_all = y_all[1:]      # (559, 50)
#         y_predict_all = y_predict_all[1:]     # (559, 50)
#
#         # 预测曲线对比，选取某一天
#         predict_curve(y_all[:271], y_predict_all[:271], x_date_5_28, id_station=id_station)      # 5月28日
#         predict_curve(y_all[271:], y_predict_all[271:], x_date_5_31, id_station=id_station)      # 5月31日
#
#         return MAE, MAPE, RMSE

# def evaluate_metric(model, data_iter, scaler, edge_index, edge_weight, device):
#     y_all = np.empty(shape=(1,10))
#     y_predict_all = np.empty(shape=(1,10))
#     model.eval()
#     with torch.no_grad():
#         mae, mape, mse = [], [], []
#         for x, y in data_iter:
#             y = scaler.inverse_transform(y.cpu().numpy())      #  (32, 50)
#             y_all = np.concatenate((y_all, y),axis=0)      # (559, 50)
#             y_pred = scaler.inverse_transform(model(x.to(device), edge_index, edge_weight).view(len(x), -1).cpu().numpy())    #  (32, 50)
#             y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)    #  (559, 50)
#
#             d = np.abs(y - y_pred)
#             mae += d.tolist()
#             mape += (d / y).tolist()
#             mse += (d ** 2).tolist()
#
#         MAE = np.array(mae).mean()
#         MAPE = np.array(mape).mean()
#         RMSE = np.sqrt(np.array(mse).mean())
#
#         # 去掉都是0的第一列
#         y_all = y_all[1:]      # (559, 50)
#         y_predict_all = y_predict_all[1:]     # (559, 50)
#
#         # 预测曲线对比，选取某一天
#         predict_curve(y_all[:271], y_predict_all[:271], x_date_5_28, id_station=0)    # 5月28日
#         predict_curve(y_all[271:], y_predict_all[271:], x_date_5_31, id_station=0)  # 5月31日
#
#         return MAE, MAPE, RMSE

# def evaluate_metric_gcn(model, W, data_iter, mean, std, id_station, if_draw):    #  [559, 50]  [559, 12, 50, 1]
#     y_all = np.empty(shape=(1,num_stations))
#     y_predict_all = np.empty(shape=(1,num_stations))
#     model.eval()
#     with torch.no_grad():
#         mae, mape, mse = [], [], []
#         for x, y in data_iter:                                                   #    [32, 12, 50, 1]  (32, 50)
#             y = scaler.inverse_transform(y.cpu().numpy())                        # (32, 50)
#             y_all = np.concatenate((y_all, y),axis=0)                            # (33, 50)
#             y_pred = scaler.inverse_transform(model(x, W).cpu().numpy())         #  (32, 50)
#
#             y_predict_all = np.concatenate((y_predict_all, y_pred),axis=0)       #  (33, 50)
#
#             d = np.abs(y - y_pred)
#             mae += d.tolist()
#             mape += masked_mape_loss(y_pred, y).tolist()
#             mse += (d ** 2).tolist()
#
#         MAE = np.array(mae).mean()
#         MAPE = np.mean(mape)
#         RMSE = np.sqrt(np.array(mse).mean())
#
#         # 去掉都是0的第一行
#         y_all = y_all[1:]                     # (1999, 50)
#         y_predict_all = y_predict_all[1:]     # (1999, 50)
#
#         # 预测曲线对比，选取某一天
#         if if_draw == True:
#             predict_curve(y_all, y_predict_all, x_date_30_gcn, id_station=id_station)
#
#         return MAE, MAPE, RMSE



# p1.plot(x, y3, marker='o', linestyle="--", color=colors1[0], linewidth=2.0, markersize=7, label='SGTGCN')
# p1.plot(x, y2, marker='*', linestyle="--", color=colors1[1], linewidth=2.0, markersize=7, label='PI-SGTGCN-noGRU')
# p1.plot(x, y1, marker='v', linestyle="--",color=colors1[2],linewidth=2.0, markersize=7, label='PI-SGTGCN-noTrans')
#
# # 给坐标轴加上标签
# plt.xlabel('Prediction Interval(mins)',fontsize=18,fontweight='bold')
# plt.ylabel('MAE reduction(%)',fontsize=18,fontweight='bold')
# plt.xticks(x, size=15, weight='bold')
# plt.yticks(size=15, weight='bold')

#
# y1 = [3.05, 1.95, 1.62, 1.85]             #  PI-SGTGCN相较于no-Trans的提升百分比
# y2 = [3.32, 3.50, 3.82, 5.32]      #  PI-SGTGCN相较于no-Gru的提升百分比
# y3 = [5.15, 2.7, 2.34, 2.33]    #  PI-SGTGCN相较于SGTGCN的提升百分比