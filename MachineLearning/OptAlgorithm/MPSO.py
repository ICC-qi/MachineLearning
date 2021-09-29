# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import numpy as np
import random
# from OptAlgorithm. PSO import PSO
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_train_patterns_test():
    # datacsv = pd.read_csv("Data_pswarm.csv", delimiter=",")
    pattern_pos = np.array([[2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0],
                            [-2.0, 2.0], [-2.0, 3.0], [-3.0, 2.0], [-3.0, 3.0],
                            [-2.0, -2.0], [-2.0, -3.0], [-3.0, -2.0], [-3.0, -3.0],
                            [2.0, -2.0], [2.0, -3.0], [3.0, -2.0], [3.0, -3.0],
                            [2.2, 2.2], [2.2, 3.2], [3.2, 2.2], [3.2, 3.2],
                            [-2.2, 2.2], [-2.2, 3.2], [-3.2, 2.2], [-3.2, 3.2],
                            [-2.2, -2.2], [-2.2, -3.2], [-3.2, -2.2], [-3.2, -3.2],
                            [2.2, -2.2], [2.2, -3.2], [3.2, -2.2], [3.2, -3.2],
                            [2.4, 2.4], [2.4, 3.4], [3.4, 2.4], [3.4, 3.4],
                            [-2.4, 2.4], [-2.4, 3.4], [-3.4, 2.4], [-3.4, 3.4],
                            [-2.4, -2.4], [-2.4, -3.4], [-3.4, -2.4], [-3.4, -3.4],
                            [2.4, -2.4], [2.4, -3.4], [3.4, -2.4], [3.4, -3.4],
                            [2.5, 2.5], [2.5, 3.5], [3.5, 2.5], [3.5, 3.5],
                            [-2.5, 2.5], [-2.5, 3.5], [-3.5, 2.5], [-3.5, 3.5],
                            [-2.5, -2.5], [-2.5, -3.5], [-3.5, -2.5], [-3.5, -3.5],
                            [2.5, -2.5], [2.5, -3.5], [3.5, -2.5], [3.5, -3.5]
                            ])
    pattern_label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                     0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    dim = 2
    class_num = 2
    print('pattern_label.shape')
    print(len(pattern_label), len(pattern_pos))
    patterns_num = len(pattern_label)
    # pattern_pos, pattern_label = x_data, y_label
    xTrain, xTest, yTrain, yTest = train_test_split(pattern_pos, pattern_label, test_size=0.3, random_state=340)
    # print(xTrain,xTest,yTrain, yTest)
    print('yTrain:')
    print(yTrain)
    print('yTest:')
    print(yTest)
    #print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
    return xTrain, xTest, yTrain, yTest, dim, class_num, patterns_num
    #return pattern_pos, pattern_label, dim, class_num, patterns_num


def load_train_patterns():
    datacsv = pd.read_csv("Data_pswarm.csv", delimiter=",")
    datacsv.head()
    type(datacsv)
    datacsv1 = datacsv[datacsv.category == 'Baby Powder']
    # Six Product Categories in the data
    # Categories are: Baby Powder, Detergent Powder, Single Serving Coffee, Toothpaste, Snacks, Canned Tuna
    print('datacsv1.shape')
    print(datacsv1.shape)
    # Six Product Categories in the data
    # Categories are: Baby Powder, Detergent Powder, Single Serving Coffee, Toothpaste, Snacks, Canned Tuna
    x = datacsv1[['Pres_Child_yes', 'Race_Caucasian', 'Ethnicity_non_Hisp', 'HH_Size_3', 'HH_Size_4',
                  'Affluence_Well', 'Affluence_Comfortable', 'HHinc_25_49k', 'HHinc_50_69k', 'HHinc_70_99k',
                  'HHinc_100k_plus', 'Age_Head_35_44Yrs', 'Age_Head_45_49Yrs', 'Age_Head_50_54Yrs', 'Age_Head_55_64Yrs',
                  'Age_Head_65plus', 'Educ_Fem_Some_College', 'Educ_Fem_Grad_College', 'Educ_Fem_Post_College',
                  'Census_East', 'Census_South', 'Census_West']]
    y = datacsv1[['Buy']]
    x_data = np.array(x)
    y_label = np.array(y).T[0]
    dim = x_data.shape[1]
    class_num = 2
    patterns_num = x_data.shape[0]
    # pattern_pos, pattern_label = x_data, y_label
    xTrain, xTest, yTrain, yTest = train_test_split(x_data, y_label, test_size=0.3, random_state=340)
    # print(xTrain,xTest,yTrain, yTest)
    print('yTrain:')
    print(yTrain)
    print('yTest:')
    print(yTest)
    print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
    return xTrain, xTest, yTrain, yTest, dim, class_num, patterns_num


def fit_fun(X, X_label, part_i, pattern_pos, pattern_label, pattern_class_particleID, patterns_num):  # 适应函数
    # for i in range(len(pattern_class_particleID)):
    g = []  # right pattern set
    b = []  # wrong pattern set
    index = [i for i, k in enumerate(pattern_class_particleID) if k == part_i]
    #print(index)  # eg [2, 5, 8] 第2，5，8个pattern是由该粒子分类的
    if index:
        for j in index:
            j_label = pattern_label[j]  # 该pattern的类别
            j_pos = pattern_pos[j]  # 该pattern的位置
            Euc_dis = np.sqrt(np.sum(np.square(j_pos - X)))  # 欧式距离
            f = 1 / (1.0 + Euc_dis)
            if j_label == X_label:
                g.append(f)
            else:
                b.append(f)
        if g:
            if b:  # 有对有错
                fitness_value = (sum(g) - sum(b)) / (sum(g) + sum(b)) + 1.0
            else:  # 全对
                fitness_value = sum(g) / patterns_num + 2.0
        else:  # 没有对的
            fitness_value = 0.0
    else:  # 没有分类的pattern
        fitness_value = 0.0
    return fitness_value

    # return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim, size, class_num, index):
        self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        # self.__fitnessValue = fit_fun(self.__pos, pattern_pos, pattern_label)  # 适应度函数值
        self.__fitnessValue = 0.0
        self.cls = math.floor(index / size)  # 粒子所属的类别

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
    def __init__(self, dim, size, class_num, iter_num, x_max, max_vel, pattern_pos, pattern_label, patterns_num,
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
        self.precision_list = []
        self.recall_list = []
        self.F1score_list = []
        self.evaluation = 0  # 准确率
        self.recall = 0
        self.F1score = 0
        self.precision = 0
        self.pattern_pos = pattern_pos
        self.pattern_label = pattern_label
        self.patterns_num = patterns_num  # patterns的数量
        self.pattern_class_particleID = [0 for i in range(len(pattern_label))]
        self.pattern_class_particleID_best = [0 for i in range(len(pattern_label))]
        self.part_best_pos = np.zeros((self.size * self.class_num, self.dim))  # 存储最好的粒子位置, 20*2

        # 对种群进行初始化
        # self.Particle_list = np.zeros((self.class_num, self.size, self.dim))
        # for ci in range(self.class_num):
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
            # vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
            # + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (
                        part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * np.sign(att_pos[i] - part.get_pos()[i]) * social_factor \
                        + self.C3 * random.random() * np.sign(part.get_pos()[i] - reg_pos[i]) * social_factor

            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part, part_i):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = fit_fun(part.get_pos(), part.cls, part_i, self.pattern_pos, self.pattern_label,
                        self.pattern_class_particleID, self.patterns_num)
        # 存储粒子最佳位置
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        '''  
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])'''

    def check_pattern_class(self):
        count = 0
        sum_num = len(self.pattern_label)
        TP = 0  # 预测为1 实际为1
        TN = 0  # 预测为0 实际为0
        FP = 0  # 预测为1 实际为0
        FN = 0  # 预测为0 实际为1
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
            pattern_i_label = self.pattern_label[p_i]  # 实际label
            part_i_cls = self.Particle_list[self.pattern_class_particleID[p_i]].cls  # 预测label
            if pattern_i_label == part_i_cls:  # correct
                count += 1
            if part_i_cls == 1:
                if pattern_i_label == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if pattern_i_label == 1:
                    FN += 1
                else:
                    TN += 1
        print('TP: ' + str(TP))
        print('TN: ' + str(TN))
        print('FP: ' + str(FP))
        print('FN: ' + str(FN))
        print('------')
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if TP == 0 and FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        #print('precision: '+str(precision))
        #print('recall: '+str(recall))
        if precision == 0 and recall == 0:
            F1score = 0
        else:
            F1score = 2 * (precision * recall) / (precision + recall)

        return count / sum_num, precision, recall, F1score

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
        self.evaluation, self.precision, self.recall, self.F1score = self.check_pattern_class()
        best_eval = self.evaluation
        for i in range(self.iter_num):
            # self.evaluation = self.check_pattern_class()
            part_x_0 = []
            part_y_0 = []
            part_x_1 = []
            part_y_1 = []
            print('iteration: ' + str(i))
            if self.evaluation > 0.99:
                break
            else:
                # for part in self.Particle_list:
                for part_i in range(self.size * self.class_num):
                    part = self.Particle_list[part_i]
                    now_cls = part.cls
                    value = fit_fun(part.get_pos(), part.cls, part_i, self.pattern_pos, self.pattern_label,
                                    self.pattern_class_particleID, self.patterns_num)
                    social_factor = 1 / (value + 1.0)
                    att_pos = self.find_close_att(part, now_cls)
                    reg_pos = self.find_close_reg(part, now_cls)
                    self.update_vel(part, att_pos, reg_pos, social_factor)  # 更新速度
                    self.update_pos(part, part_i)  # 更新位置
                self.evaluation, self.precision, self.recall, self.F1score = self.check_pattern_class()  # 更新完算准确率
                # self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
                self.fitness_val_list.append(self.evaluation)
                self.precision_list.append(self.precision)
                self.recall_list.append(self.recall)
                self.F1score_list.append(self.F1score)
                # self.evaluation = 0
                if self.evaluation > best_eval:  # 当前最好的粒子位置，以准确率最高的原则判定的
                    best_eval = self.evaluation
                    self.pattern_class_particleID_best = self.pattern_class_particleID.copy()
                    for particle_i in range(self.size * self.class_num):
                        particle = self.Particle_list[particle_i]
                        part_pos = np.array(particle.get_pos())
                        self.part_best_pos[particle_i] = part_pos
                        if particle_i < self.size * self.class_num / 2:
                            part_x_0.append(part_pos[0])
                            part_y_0.append(part_pos[1])
                        else:
                            part_x_1.append(part_pos[0])
                            part_y_1.append(part_pos[1])
                    #plt.figure()
                    #plt.scatter(part_x_0, part_y_0, c='r', alpha=0.5)
                    #plt.scatter(part_x_1, part_y_1, c='g', alpha=0.5)
                    #plt.show()
        return self.part_best_pos, self.fitness_val_list, self.precision_list, self.recall_list, self.F1score_list  # , self.get_bestPosition()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    # 加载数据 load data
    #pattern_pos, pattern_label, dim, class_num, patterns_num = load_train_patterns_test()
    xTrain, xTest, yTrain, yTest, dim, class_num, patterns_num = load_train_patterns_test()  # 64点的测试demo
    #xTrain, xTest, yTrain, yTest, dim, class_num, patterns_num = load_train_patterns()  # 真实数据
    # print(xTrain.shape, xTest.shape,yTrain.shape, yTest.shape, dim, class_num, patterns_num)

    # Data Standardization 数据标准化
    xTrainfit = preprocessing.StandardScaler().fit(xTrain).transform(xTrain.astype(float))
    xTestfit = preprocessing.StandardScaler().fit(xTest).transform(xTest.astype(float))
    #print(xTrainfit)

    # hyper-parameters 设置超参
    # dim = 2
    size = 7  # 每类粒子个数
    iter_num = 100
    x_max = 2
    max_vel = 4  # 最大速度：默认0.5

    # MPSO粒子初始化
    #pso = PSO(dim, size, class_num, iter_num, x_max, max_vel, pattern_pos, pattern_label, patterns_num)
    pso = PSO(dim, size, class_num, iter_num, x_max, max_vel, xTrainfit, yTrain, len(yTrain))
    pso_test = PSO(dim, size, class_num, iter_num, x_max, max_vel, xTestfit, yTest, len(yTest))

    # 训练 原来的标准PSO代码 training the code of PSO
    # fit_var_list, best_pos = pso.update()
    # print("最优位置:" + str(best_pos))
    # print("最优解:" + str(fit_var_list[-1]))

    # MPSO训练 MPSO training
    part_best_pos, fit_var_list, train_precision_list, train_recall_list, train_F1score_list = pso.update()
    print('training finish')
    test_evaluation, test_precision, test_recall, test_F1score = pso_test.check_pattern_class()
    print('training set evaluation:')
    print(max(fit_var_list))
    print('training precision:')
    print(max(train_precision_list))
    print('training recall:')
    print(max(train_recall_list))
    print('training F1score:')
    print(max(train_F1score_list))
    plt.figure()
    #plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list, c='r', alpha=0.5)
    plt.plot(np.linspace(0, len(fit_var_list), len(fit_var_list)), fit_var_list, c='r', alpha=0.5)
    plt.show()
    print('best particles position:')
    print(part_best_pos)
    print('test set evaluation:')
    print(test_evaluation)
    print('test precision:')
    print(test_precision)
    print('test recall:')
    print(test_recall)
    print('test F1score:')
    print(test_F1score)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

