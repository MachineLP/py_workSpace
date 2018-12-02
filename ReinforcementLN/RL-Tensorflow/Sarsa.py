import numpy as np
import pandas as pd
import time
np.random.seed(2)  # reproducible
# 定义6种状态
N_STATES = 6   
# 在线性状态下只能采取往左或者往右
ACTIONS = ['left', 'right']     
EPSILON = 0.9   # greedy police
# 学习率
ALPHA = 0.1   
# 随机因素， 我们有10%的可能随便选取行动  
GAMMA = 0.9    
# 我们的智能体， 进化次数
MAX_EPISODES = 13   
# 为了防止太快，方便观看，sleep一下
FRESH_TIME = 0.3   
# Q表用来记录每种状态采取的行动回报值。
# 下面是进行初始化；
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),   # 初始化奖励值
        columns=actions,   # 采取的行动
    )
    # print(table)    
    return table
# 根据Q表，获取目前状态下采取的行动， 注意有10%的随机性
def choose_action(state, q_table):
    # 获得在某状态下的奖励， 但是如何行动的话，怎么选择？
    # 两种方式：（1）10%的随机；（2）选取回报最大的作为下一步的行动；
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): 
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    
    return action_name
# 根据行动后，所在的状态给予奖励；
def get_env_feedback(S, A):
    # 这里智能体就会得到反馈；
    # 往右移动
    if A == 'right':   
        # 这就是延时回报的原因，开始进化时只有到了最后我们才知道是否应该给予奖励
        if S == N_STATES - 2:   
            S_ = 'terminal'
            R = 1
        # 下面虽然没有给予奖励，但是状态加一，也就是说目的地更近了一步，也算是一种奖励
        else:
            S_ = S + 1
            R = 0
    # 那么如果你往左，下面都是惩罚
    else:   
        R = 0
        if S == 0:
            S_ = S 
        else:
            S_ = S - 1
    return S_, R
# 用来更新目前的结果 和 现实
def update_env(S, episode, step_counter):
    env_list = ['>>>']*(N_STATES-1) + ['OK'] 
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'ooo'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
# 下面就是智能体核心进化流程， 也就是一个算法的优化流程；
def rl():
    # 初始化Q表
    q_table = build_q_table(N_STATES, ACTIONS)
    # 智能体进化次数
    for episode in range(MAX_EPISODES):
        step_counter = 0
        # 状态从0开始；
        S = 0
        # 行动往左开始；
        A = 'left'
        # 一个标示， 表示是否到达终点。
        is_terminated = False
        # 更新显示
        update_env(S, episode, step_counter)
        # 如果智能体没有到达目的地， 不停的迭代
        while not is_terminated:
            # 根据此时状态和采取的行动， 得到下一个所在的状态和应得奖励
            S_, R = get_env_feedback(S, A)  
            # 判断上面采取行动A后是否到达目的地； 如果没有，此时再此状态从Q表获得下一步的行动；
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
            # 获得S状态A行动下的回报值，这里是后面此时Q表的更新；
            q_predict = q_table.loc[S, A]
            # Sarsa算法的精髓
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.loc[S_, A_] #.max()   # next state is not terminal
            # 达到目的地获得奖励， 回报给上一个状态动作哦， 就是这样回传的。
            else:
                q_target = R     
                is_terminated = True    
            # 更新
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
            S = S_  
            A = A_
            print (S, A)
            update_env(S, episode, step_counter+1)
            step_counter += 1
            print (q_table)
    return q_table
if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
--------------------- 
作者：MachineLP 
来源：CSDN 
原文：https://blog.csdn.net/u014365862/article/details/79240997 
版权声明：本文为博主原创文章，转载请附上博文链接！
