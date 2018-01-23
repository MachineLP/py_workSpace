

from keras.datasets import mnist
from keras.layers import *
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import keras
import keras.backend as K

import numpy as np

from sklearn.metrics import accuracy_score
import sklearn

import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm import tqdm
from collections import deque
import seaborn as sns
import random,time

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_actions = len(set(y_test))
image_w, image_h = X_train.shape[1:]
print (image_w, image_h)

X_train = X_train.reshape(*X_train.shape,1)
X_test = X_test.reshape(*X_test.shape, 1)

#normalization
X_train = X_train/255.
X_test = X_test/255.

dummy_actions = np.ones((1, num_actions))

# 转为one-hot
y_train_onehot = keras.utils.to_categorical(y_train, num_actions)
y_test_onehot = keras.utils.to_categorical(y_test, num_actions)

plt.imshow(X_train[0].reshape(28,28),'gray')

# 强化学习的过程;
# agent、环境状态、行动、奖励;
class MnEnviroment(object):
    def __init__(self, x,y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self._sample_index()
        self.action_space = len(set(y)) - 1
    def reset(self):
        obs, _ = self.step(-1)
        return obs
    '''
    action: 0-9 categori, -1 : start and no reward
    return: next_state(image), reward
    '''
    def step(self, action):
        if action==-1:
            _c_index = self.current_index
            self.current_index = self._sample_index()
            return (self.train_X[_c_index], 0)
        r = self.reward(action)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index], r
    
    def reward(self, action):
        c = self.train_Y[self.current_index]
        #print(c)
        return 1 if c==action else -1
        
    def sample_actions(self):
        return random.randint(0, self.action_space)
    
    def _sample_index(self):
        return random.randint(0, len(self.train_Y)-1)


# 定义环境状态
env = MnEnviroment(X_train, y_train)

memory = deque(maxlen=512)
replay_size = 64
epoches = 2000
pre_train_num = 256
gamma = 0.  #every state is i.i.d
alpha = 0.5
forward = 512
epislon_total = 2018



def createDQN(input_width, input_height, actions_num):
    img_input = Input(shape=(input_width, input_height,1),dtype='float32',name='image_inputs')
    #conv1
    conv1 = Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(img_input)
    conv2 = Conv2D(64,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(64,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(128,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv3)
    x = Flatten()(conv4)
    x = Dense(128, activation='relu')(x)
    outputs_q = Dense(actions_num, name='q_outputs')(x)
    #one hot input
    actions_input = Input((actions_num,),name='actions_input')
    q_value= multiply([actions_input, outputs_q])
    q_value = Lambda(lambda l:K.sum(l, axis=1,keepdims=True),name='q_value')(q_value)
    
    model = Model(inputs=[img_input, actions_input], outputs=q_value)
    model.compile(loss='mse',optimizer='adam')
    return model

actor_model = createDQN(image_w,image_h,num_actions) #用于决策
critic_model = createDQN(image_w,image_h,num_actions) #用于训练
actor_q_model = Model(inputs=actor_model.input, outputs=actor_model.get_layer('q_outputs').output)


def copy_critic_to_actor():
    critic_weights = critic_model.get_weights()
    actor_wegiths  = actor_model.get_weights()
    for i in range(len(critic_weights)):
        actor_wegiths[i] = critic_weights[i]
    actor_model.set_weights(actor_wegiths)


def get_q_values(model_,state):
    inputs_ = [state.reshape(1,*state.shape),dummy_actions]
    qvalues = model_.predict(inputs_)
    return qvalues[0]

def predict(model,states):
    inputs_ = [states, np.ones(shape=(len(states),num_actions))]
    qvalues = model.predict(inputs_)
    return np.argmax(qvalues,axis=1)

def epsilon_calc(step, ep_min=0.01,ep_max=1,ep_decay=0.0001,esp_total = 1000):
    return max(ep_min, ep_max -(ep_max - ep_min)*step/esp_total )

def epsilon_greedy(env, state, step, ep_min=0.01, ep_decay=0.0001,ep_total=1000):
    epsilon = epsilon_calc(step, ep_min, 1, ep_decay,ep_total)
    if np.random.rand()<epsilon:
        return env.sample_actions(),0
    qvalues = get_q_values(actor_q_model, state)
    return np.argmax(qvalues), np.max(qvalues)

def pre_remember(pre_go = 30):
    state = env.reset()
    for i in range(pre_go):
        rd_action = env.sample_actions()
        next_state, reward = env.step(rd_action)
        remember(state,rd_action,0,reward,next_state)
        state = next_state

def remember(state,action,action_q,reward,next_state):
    memory.append([state,action,action_q,reward,next_state])
    
def sample_ram(sample_num):
    return np.array(random.sample(memory,sample_num))

def replay():
    if len(memory) < replay_size:
        return 
    #从记忆中i.i.d采样
    samples = sample_ram(replay_size)
    #展开所有样本的相关数据
    #这里next_states没用 因为和上一个state无关。
    states, actions, old_q, rewards, next_states = zip(*samples)
    states, actions, old_q, rewards = np.array(states),np.array(actions).reshape(-1,1),\
                                    np.array(old_q).reshape(-1,1),np.array(rewards).reshape(-1,1)
   
    actions_one_hot = np_utils.to_categorical(actions,num_actions)
    #print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
    #从actor获取下一个状态的q估计值 这里也没用 因为gamma=0 也就是不对bellman方程展开
    #inputs_ = [next_states,np.ones((replay_size,num_actions))]
    #qvalues = actor_q_model.predict(inputs_)
    
    #q = np.max(qvalues,axis=1,keepdims=True)
    q = 0
    q_estimate = (1-alpha)*old_q +  alpha *(rewards.reshape(-1,1) + gamma * q)
    history = critic_model.fit([states,actions_one_hot],q_estimate,epochs=1,verbose=0)
    return np.mean(history.history['loss'])

memory.clear()
total_rewards = 0
reward_rec = []
pre_remember(pre_train_num)
every_copy_step = 128

epoches, forward, epislon_total,pre_train_num

pbar = tqdm(range(1,epoches+1))
state = env.reset()
for epoch in pbar:
    total_rewards = 0
    epo_start = time.time()
    for step in range(forward):
        #对每个状态使用epsilon_greedy选择
        action,q = epsilon_greedy(env, state, epoch, ep_min=0.01, ep_total=epislon_total)
        eps = epsilon_calc(epoch,esp_total=epislon_total)
        #play 
        next_state,reward = env.step(action)
        #加入到经验记忆中
        remember(state, action, q, reward, next_state)
        #从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
        loss = replay()
        total_rewards += reward
        state = next_state
        if step % every_copy_step==0:
            copy_critic_to_actor()
    reward_rec.append(total_rewards)
    pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards,loss,int(time.time()-epo_start),eps))

