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
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

