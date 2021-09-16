# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import numpy as np
import random
#from OptAlgorithm. PSO import PSO
import matplotlib. pyplot as plt
import numpy as np
import math
#import pandas as pd
import os
import sys

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def load_train_patterns():
    #datacsv = pd.read_csv("Data_pswarm.csv", delimiter=",")
    pattern_pos = np.array([[2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0],
                            [-2.0, 2.0], [-2.0, 3.0], [-3.0, 2.0], [-3.0, 3.0],
                            [-2.0, -2.0], [-2.0, -3.0], [-3.0, -2.0], [-3.0, -3.0],
                            [2.0, -2.0], [2.0, -3.0], [3.0, -2.0], [3.0, -3.0],
                            [0.5, -0.5], [-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
                            ])
    pattern_label = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    dim = 2
    class_num = 2
    return pattern_pos, pattern_label, dim, class_num


def fit_fun(X):  # 适应函数
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim, size, class_num, index):
        self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值
        self.cls = math.floor(index/size)  # 粒子所属的类别

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, class_num, iter_num, x_max, max_vel,  pattern_pos, pattern_label,
                 best_fitness_value=float('Inf'), C1=2, C2=2, C3=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.class_num = class_num  # 类的个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.evaluation = 0  # 准确率
        self.pattern_pos = pattern_pos
        self.pattern_label = pattern_label
        self.pattern_class_particleID = [0 for i in range(len(pattern_label))]
        self.pattern_class_particleID_best = [0 for i in range(len(pattern_label))]
        self.part_best_pos = np.zeros((self.size*self.class_num, self.dim))  # 存储最好的粒子位置, 20*2


        # 对种群进行初始化
        #self.Particle_list = np.zeros((self.class_num, self.size, self.dim))
        #for ci in range(self.class_num):
        #    for i in range(self.size):
        #        self.Particle_list[ci][i] = Particle(self.x_max, self.max_vel, self.dim)
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, self.size,
                                       self.class_num, i) for i in range(self.size * self.class_num)]
        print('a')

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part, att_pos, reg_pos, social_factor):
        for i in range(self.dim):
            #vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        #+ self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * np.sign(att_pos[i] - part.get_pos()[i]) * social_factor \
                        + self.C3 * random.random() * np.sign(part.get_pos()[i] - reg_pos[i]) * social_factor

            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = fit_fun(part.get_pos())
        '''  # 存储粒子最佳位置
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])'''

    def check_pattern_class(self):
        count = 0
        sum_num = len(self.pattern_label)
        for p_i in range(sum_num):
            min_Euc_dis = 100.0
            for i in range(self.size * self.class_num):
                part = self.Particle_list[i]
                part_pos = np.array(part.get_pos())
                pattern_pos = self.pattern_pos[p_i]
                Euc_dis = np.sqrt(np.sum(np.square(part_pos - pattern_pos)))
                if Euc_dis < min_Euc_dis:
                    min_Euc_dis = Euc_dis
                    self.pattern_class_particleID[p_i] = i
            pattern_i_label = self.pattern_label[p_i]
            part_i_cls = self.Particle_list[self.pattern_class_particleID[p_i]].cls
            if pattern_i_label == part_i_cls:  # correct
                count += 1
        return count/sum_num


    def find_close_att(self, part, now_cls):  # 不是同一类的吸引
        min_Euc_dis = 100.0
        att_pos = part.get_pos()
        for i in range(len(self.pattern_class_particleID)):
            pattern_i_cls = self.pattern_label[i]
            if now_cls == pattern_i_cls:
                class_part_index = self.pattern_class_particleID[i]
                class_part = self.Particle_list[class_part_index]
                class_part_cls = class_part.cls
                if not class_part_cls == now_cls:
                    class_part_pos = np.array(class_part.get_pos())
                    Euc_dis = np.sqrt(np.sum(np.square(class_part_pos - np.array(part.get_pos()))))
                    if Euc_dis < min_Euc_dis:
                        min_Euc_dis = Euc_dis
                        att_pos = class_part_pos
            else:
                continue
        return att_pos


    def find_close_reg(self, part, now_cls):  # 同一类的排斥
        min_Euc_dis = 100.0
        reg_pos = part.get_pos()
        for i in range(len(self.pattern_class_particleID)):
            pattern_i_cls = self.pattern_label[i]
            if now_cls == pattern_i_cls:
                class_part_index = self.pattern_class_particleID[i]
                class_part = self.Particle_list[class_part_index]
                class_part_cls = class_part.cls
                if class_part_cls == now_cls:
                    class_part_pos = np.array(class_part.get_pos())
                    Euc_dis = np.sqrt(np.sum(np.square(class_part_pos - np.array(part.get_pos()))))
                    if Euc_dis < min_Euc_dis:
                        min_Euc_dis = Euc_dis
                        reg_pos = class_part_pos
            else:
                continue
        return reg_pos

    def update(self):
        self.evaluation = self.check_pattern_class()
        best_eval = self.evaluation
        for i in range(self.iter_num):
            #self.evaluation = self.check_pattern_class()
            part_x_0 = []
            part_y_0 = []
            part_x_1 = []
            part_y_1 = []
            if self.evaluation > 0.99:
                break
            else:
                for part in self.Particle_list:
                    now_cls = part.cls
                    value = fit_fun(part.get_pos())
                    social_factor = 1/(value+1.0)
                    att_pos = self.find_close_att(part, now_cls)
                    reg_pos = self.find_close_reg(part, now_cls)
                    self.update_vel(part, att_pos, reg_pos, social_factor)  # 更新速度
                    self.update_pos(part)  # 更新位置
                self.evaluation = self.check_pattern_class()  # 更新完算准确率
                #self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
                self.fitness_val_list.append(self.evaluation)
                #self.evaluation = 0
                if self.evaluation > best_eval:  # 当前最好的粒子位置
                    best_eval = self.evaluation
                    self.pattern_class_particleID_best = self.pattern_class_particleID.copy()
                    for particle_i in range(self.size*self.class_num):
                        particle = self.Particle_list[particle_i]
                        part_pos = np.array(particle.get_pos())
                        self.part_best_pos[particle_i] = part_pos
                        if particle_i < self.size*self.class_num/2:
                            part_x_0.append(part_pos[0])
                            part_y_0.append(part_pos[1])
                        else:
                            part_x_1.append(part_pos[0])
                            part_y_1.append(part_pos[1])
                    plt.figure()
                    plt.scatter(part_x_0, part_y_0, c='r', alpha=0.5)
                    plt.scatter(part_x_1, part_y_1, c='g', alpha=0.5)
                    plt.show()
        return self.part_best_pos, self.fitness_val_list#, self.get_bestPosition()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    pattern_pos, pattern_label, dim, class_num = load_train_patterns()
    #dim = 2
    size = 8  # 每类粒子个数
    iter_num = 1000
    x_max = 5
    max_vel = 2  # 0.5
    pso = PSO(dim, size, class_num, iter_num, x_max, max_vel, pattern_pos, pattern_label)
    #fit_var_list, best_pos = pso.update()
    #print("最优位置:" + str(best_pos))
    #print("最优解:" + str(fit_var_list[-1]))
    part_best_pos, fit_var_list = pso.update()
    plt.figure()
    #plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list, c='r', alpha=0.5)
    plt.plot(np.linspace(0, len(fit_var_list), len(fit_var_list)), fit_var_list, c='r', alpha=0.5)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

