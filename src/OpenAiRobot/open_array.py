import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime


p = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/AContinuousCartPole_discount_809599'
p = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/figures'
dirs80 = [x[0] for x in os.walk(p + '/80')]
dirs95 = [x[0] for x in os.walk(p + '/90')]
dirs99 = [x[0] for x in os.walk(p + '/99')]

arrs80 = []
arrs95 = []
arrs99 = []

for p in dirs80[1:]:
    temp_arr = np.load(p + '/score_array.npy')
    arrs80.append(temp_arr)

for p in dirs95[1:]:
    temp_arr = np.load(p + '/score_array.npy')
    arrs95.append(temp_arr)
# a = np.load('/home/emilal/Documents/RLRobotics/src/OpenAiRobot/figures/constructed/score_array.npy')
for p in dirs99[1:]:
    temp_arr = np.load(p + '/score_array.npy')
    arrs99.append(temp_arr)

save_path = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/figures/constructed'

var80 = np.array(arrs80).std(axis=0)
mean80 = np.array(arrs80).mean(axis=0)
var95 = np.array(arrs95).std(axis=0)
mean95 = np.array(arrs95).mean(axis=0)
var99 = np.array(arrs99).std(axis=0)
mean99 = np.array(arrs99).mean(axis=0)
#plt.ioff()
# plt.figure(figsize=(12,12))
plt.figure(figsize=(10,5))
# plt.title('Continuous Cart Pole')
plt.title('Pendulum')
# fig, (ax1, ax2) = plt.subplot(2,1)
plt.fill_between(range(1901), mean80+var80/2, mean80-var80/2, alpha=0.5, label='discount 0.80')
plt.fill_between(range(1901), mean95+var95/2, mean95-var95/2, alpha=0.5, label='discount 0.95')
plt.fill_between(range(1901), mean99+var99/2, mean99-var99/2, alpha=0.5, label='discount 0.99')
plt.plot()
# plt.plot(np.array(arrs80).T, 'r', np.array(arrs95).T, 'xkcd:sky blue',np.array(arrs99).T, 'm')
# plt.plot(np.array(arrs80).T, 'b', np.array(arrs99).T, 'r')

# #trippleplots
# plt.subplot(3,1,1)
# plt.plot(np.array(arrs80).T)
# plt.ylabel('reward')
# plt.title('Rewards for discount = 0.80')
# plt.subplot(3,1,2)
# plt.plot(np.array(arrs95).T)
# plt.ylabel('reward')
# plt.title('Rewards for discount = 0.90')
# plt.subplot(3,1,3)
# plt.plot(np.array(arrs99).T)
# plt.ylabel('reward')
# plt.title('Rewards for discount = 0.99')
# # tripleplots

# leg = plt.legend(["discount 0.80"], loc='upper left')
# for item in leg.legendHandles:
#     item.set_visible(False)
# plt.subplot(2,1,2)
# plt.plot(np.array(arrs95).T)
# leg = plt.legend(["discount 0.99"], loc='upper left')
# for item in leg.legendHandles:
#     item.set_visible(False)
# plt.errorbar(range(500),y=mean80,yerr=var80/2)
# plt.errorbar(range(500),y=mean95,yerr=var95/2)
# plt.errorbar(range(500),y=mean99,yerr=var99/2)
#plt.annotate(info, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
# plt.xlabel('epoch [{} games]'.format(100))
plt.xlabel('episode [200 timesteps]'.format(100))
plt.ylabel('reward')

plt.legend(loc='upper left')

plt.savefig(save_path + '/discCP8090_{}'.format(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
plt.clf()

# p = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/DiscreteCartPoleWith_1_8/2017-12-06_12_25_22'

# p = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/mounainCartContActorCritic/NiceOnes/'
#
# save_path = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/figures/constructed'
#
# arr1 = np.load(p + 'score_array.npy')
# arr2 = np.load(p + 'score_array2.npy')
# arr3 = np.load(p + 'score_array3.npy')
# # arr4 = np.load(p + 'score_array4.npy')
#
# #plt.ioff()
# plt.figure(figsize=(12, 6))
# # fig, (ax1) = plt.subplot(1,1)
# plt.plot(arr1, label='Run 1')
# plt.plot(arr2, label='Run 2')
# plt.plot(arr3, label='Run 3')
# # plt.errorbar(range(500),y=mean80,yerr=var80/2)
# # plt.errorbar(range(500),y=mean95,yerr=var95/2)
# # plt.errorbar(range(500),y=mean99,yerr=var99/2)
# plt.title('Rewards')
# #plt.annotate(info, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
# plt.xlabel('epoch [{} game(s)]'.format(1))
# plt.ylabel('reward')
#
# plt.legend(loc='lower right')
#
# plt.savefig(save_path + '/discount_full_{}'.format(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
# plt.clf()

# p1 ='/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/ContinuousCartRunWstddiv/2017-11-03_10_38_49/'
# p2 = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/DiscreteCartPoleWidth2_3/2017-12-05_20_29_24/'
# p3 = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/DiscreteCartPoleWidth2_3/2017-12-05_13_23_02/'
# p4 = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/DiscreteCartPoleDiscount80_95_99/0_95/2017-12-01_21_56_00/'
# p8 = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/training_log/DiscreteCartPoleWith_1_8/2017-12-06_22_32_05/'
# save_path = '/home/emilal/Documents/RLRobotics/src/OpenAiRobot/figures/constructed'
#
#
# ones = np.ones(500)
# discount99 = 0
# discount95 = 0
# discount80 = 0
# rew99 = np.zeros(len(ones))
# rew95 = np.zeros(len(ones))
# rew80 = np.zeros(len(ones))
# for elem in reversed(range(len(ones))):
#     rew99[elem] = ones[elem] + 0.99*discount99
#     discount99 = rew99[elem]
#     rew95[elem] = ones[elem] + 0.95*discount95
#     discount95 = rew95[elem]
#     rew80[elem] = ones[elem] + 0.80*discount95
#     discount80 = rew80[elem]
#
#
# # arr1 = np.load(p1 + 'score_array.npy')
# # arr1 = arr1[:1000]
# # arr2 = np.load(p2 + 'score_array.npy')
# # arr3 = np.load(p3 + 'score_array.npy')
# # arr4 = np.load(p4 + 'score_array.npy')
# # arr8 = np.load(p8 + 'score_array.npy')
#
# #plt.ioff()
# plt.figure(figsize=(10, 5))
# # fig, (ax1) = plt.subplot(1,1)
# plt.plot(rew99, 'b', label='Discount: 0.99')
# plt.plot(rew95, 'r', label='Discount: 0.95')
# plt.plot(rew80, 'g', label='Discount: 0.80')
# # plt.plot(arr4, 'k', label='width hidden layer: 4')
# # plt.plot(arr8, 'm', label='width hidden layer: 8')
# # plt.errorbar(range(500),y=mean80,yerr=var80/2)
# # plt.errorbar(range(500),y=mean95,yerr=var95/2)
# # plt.errorbar(range(500),y=mean99,yerr=var99/2)
# # plt.title('Rewards')
# plt.title('Returns')
# #plt.annotate(info, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
# # plt.xlabel('epoch [{} game(s)]'.format(100))
# plt.xlabel('timesteps [t]'.format(100))
# plt.ylabel('return')
#
# plt.legend(loc='upper right')
#
# plt.savefig(save_path + '/returns{}'.format(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
# plt.clf()
