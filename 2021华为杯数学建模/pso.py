from pyswarm import pso
import pandas as pd
import warnings 
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
from sklearn import preprocessing
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
import random
import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO
from xgboost import XGBClassifier

#导入数据，(1973,20)
# data_f = pd.read_excel('ADMET_select.xlsx',index_col=0,engine='openpyxl')
# data_label = pd.read_excel('data_label_ER.xlsx',index_col=0,engine='openpyxl')
data_f = pd.read_excel('Molecular_Descriptor_select_feature.xlsx',index_col=0,engine='openpyxl')
data_label_cls = pd.read_excel('ADMET_final_label.xlsx',index_col=0,engine='openpyxl')
data_label_reg = pd.read_excel('ER_activity.xlsx',index_col=0,engine='openpyxl')
#根据(1973,20)数据训练出一个回归器，再使用粒子群算法优化该回归器
rfr = RandomForestRegressor(n_estimators=1000,max_depth=100)
rfr.fit(data_f,data_label_reg['pIC50'])
print('随机森林回归训练完成')
xgb = XGBClassifier(n_estimators=100,gamma=0.1,max_depth=4,min_child_weight=3)
xgb.fit(data_f,data_label_cls['Fina_label'])
print('xgboost分类训练完成')
#根据（1973，20）数据训练出一个二分类器，作为约束条件



def fitness(ind_var):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = ind_var
    # x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = ind_var[0],ind_var[1],ind_var[2],ind_var[3],
    # ind_var[4],ind_var[5],ind_var[6],ind_var[7],ind_var[8],ind_var[9],ind_var[10],ind_var[11],ind_var[12],ind_var[13],
    # ind_var[14],ind_var[15],ind_var[16],ind_var[17],ind_var[18],ind_var[19]
    data_pre = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20]).reshape(1,-1)
    y = rfr.predict(data_pre)
    return -float(y)#添加负号最大化目标函数
# list(data_f.min())
#     up = list(data_f.max())
cond = (
    lambda x:xgb.predict(x)
)
print('开始执行PSO算法')
pso = PSO(func=fitness,n_dim=20,pop=60, max_iter=2, lb=list(data_f.min()), ub=list(data_f.max()), w=random.uniform(0,1), c1=0.5, c2=0.5,constraint_eq=cond)
pso.record_mode=True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
# data_log = pd.DataFrame(pso.record_value)
# data_log.to_excel('PSO_LOG.xlsx')
print(len(pso.record_value['X']))
# print(pso.record_value)
# plt.plot(pso.gbest_y_hist)
# plt.show()
# class PSO:
#     def __init__(self, parameters):
#         """
#         particle swarm optimization
#         parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
#         """
#         # 初始化
#         self.NGEN = parameters[0]    # 迭代的代数
#         self.pop_size = parameters[1]    # 种群大小
#         self.var_num = 20     # 变量个数
#         self.bound = []                 # 变量的约束范围
#         self.bound.append(parameters[2])
#         self.bound.append(parameters[3])
 
#         self.pop_x = np.zeros((self.pop_size, self.var_num))    # 所有粒子的位置
#         self.pop_v = np.zeros((self.pop_size, self.var_num))    # 所有粒子的速度
#         self.p_best = np.zeros((self.pop_size, self.var_num))   # 每个粒子最优的位置
#         self.g_best = np.zeros((1, self.var_num))   # 全局最优的位置
 
#         # 初始化第0代初始全局最优解
#         temp = -1
#         for i in range(self.pop_size):
#             for j in range(self.var_num):
#                 self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
#                 self.pop_v[i][j] = random.uniform(0, 1)
#             self.p_best[i] = self.pop_x[i]      # 储存最优的个体
#             fit = self.fitness(self.p_best[i])
#             if fit > temp:
#                 self.g_best = self.p_best[i]
#                 temp = fit
 
#     def fitness(self, ind_var):
#         """
#         个体适应值计算
#         """
# #         x1 = ind_var[0]
# #         x2 = ind_var[1]
# #         x3 = ind_var[2]
# #         x4 = ind_var[3]
# #         y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
#         x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = ind_var[0],ind_var[1],ind_var[2],ind_var[3],
#         ind_var[4],ind_var[5],ind_var[6],ind_var[7],ind_var[8],ind_var[9],ind_var[10],ind_var[11],ind_var[12],ind_var[13],
#         ind_var[14],ind_var[15],ind_var[16],ind_var[17],ind_var[18],ind_var[19]
#         y = rfr.predict(list(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20))
#         return y
 
#     def update_operator(self, pop_size):
#         """
#         更新算子：更新下一时刻的位置和速度
#         """
#         c1 = 2     # 学习因子，一般为2
#         c2 = 2
#         w = 0.4    # 自身权重因子
#         for i in range(pop_size):
#             # 更新速度
#             self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
#                         self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
#             # 更新位置
#             self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
#             # 越界保护
#             for j in range(self.var_num):
#                 if self.pop_x[i][j] < self.bound[0][j]:
#                     self.pop_x[i][j] = self.bound[0][j]
#                 if self.pop_x[i][j] > self.bound[1][j]:
#                     self.pop_x[i][j] = self.bound[1][j]
#             # 更新p_best和g_best
#             if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
#                 self.p_best[i] = self.pop_x[i]
#             if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
#                 self.g_best = self.pop_x[i]
 
#     def main(self):
#         popobj = []
#         self.ng_best = np.zeros((1, self.var_num))[0]
#         for gen in range(self.NGEN):
#             self.update_operator(self.pop_size)
#             popobj.append(self.fitness(self.g_best))
#             print('############ Generation {} ############'.format(str(gen + 1)))
#             if self.fitness(self.g_best) > self.fitness(self.ng_best):
#                 self.ng_best = self.g_best.copy()
#             print('最好的位置：{}'.format(self.ng_best))
#             print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
#         print("---- End of (successful) Searching ----")
# class PSO:
#     def __init__(self, parameters):
#         """
#         particle swarm optimization
#         parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
#         """
#         # 初始化
#         self.NGEN = parameters[0]    # 迭代的代数
#         self.pop_size = parameters[1]    # 种群大小
#         self.var_num = 20     # 变量个数
#         self.bound = []                 # 变量的约束范围
#         self.bound.append(parameters[2])
#         self.bound.append(parameters[3])
 
#         self.pop_x = np.zeros((self.pop_size, self.var_num))    # 所有粒子的位置
#         self.pop_v = np.zeros((self.pop_size, self.var_num))    # 所有粒子的速度
#         self.p_best = np.zeros((self.pop_size, self.var_num))   # 每个粒子最优的位置
#         self.g_best = np.zeros((1, self.var_num))   # 全局最优的位置
 
#         # 初始化第0代初始全局最优解
#         temp = -1
#         for i in range(self.pop_size):
#             for j in range(self.var_num):
#                 self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
#                 self.pop_v[i][j] = random.uniform(0, 1)
#             self.p_best[i] = self.pop_x[i]      # 储存最优的个体
#             fit = self.fitness(self.p_best[i])
#             if fit > temp:
#                 self.g_best = self.p_best[i]
#                 temp = fit
 
#     def fitness(self, ind_var):
#         """
#         个体适应值计算
#         """
# #         x1 = ind_var[0]
# #         x2 = ind_var[1]
# #         x3 = ind_var[2]
# #         x4 = ind_var[3]
# #         y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
#         x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = ind_var[0],ind_var[1],ind_var[2],ind_var[3],\
#         ind_var[4],ind_var[5],ind_var[6],ind_var[7],ind_var[8],ind_var[9],ind_var[10],ind_var[11],ind_var[12],ind_var[13],\
#         ind_var[14],ind_var[15],ind_var[16],ind_var[17],ind_var[18],ind_var[19]
#         y = rfr.predict(list(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20))
#         return y
 
#     def update_operator(self, pop_size):
#         """
#         更新算子：更新下一时刻的位置和速度
#         """
#         c1 = 2     # 学习因子，一般为2
#         c2 = 2
#         w = 0.4    # 自身权重因子
#         for i in range(pop_size):
#             # 更新速度
#             self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
#                         self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
#             # 更新位置
#             self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
#             # 越界保护
#             for j in range(self.var_num):
#                 if self.pop_x[i][j] < self.bound[0][j]:
#                     self.pop_x[i][j] = self.bound[0][j]
#                 if self.pop_x[i][j] > self.bound[1][j]:
#                     self.pop_x[i][j] = self.bound[1][j]
#             # 更新p_best和g_best
#             if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
#                 self.p_best[i] = self.pop_x[i]
#             if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
#                 self.g_best = self.pop_x[i]
 
#     def main(self):
#         popobj = []
#         self.ng_best = np.zeros((1, self.var_num))[0]
#         for gen in range(self.NGEN):
#             self.update_operator(self.pop_size)
#             popobj.append(self.fitness(self.g_best))
#             print('############ Generation {} ############'.format(str(gen + 1)))
#             if self.fitness(self.g_best) > self.fitness(self.ng_best):
#                 self.ng_best = self.g_best.copy()
#             print('最好的位置：{}'.format(self.ng_best))
#             print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
#         print("---- End of (successful) Searching ----")
# if __name__ == '__main__':
#     NGEN = 100
#     popsize = 20
#     low = list(data_f.min())
#     up = list(data_f.max())
#     parameters = [NGEN, popsize, low, up]
#     pso = PSO(parameters)
#     pso.main()