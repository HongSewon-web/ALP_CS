import os, sys
import numpy as np
import tensorflow as tf
#유사난수 초기화(일관된 출력)
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
os.environ["TF_USE_LEGACY_KERAS"]='True'
os.environ["TF_USE_LEGACY_KERAS"]="0"
# 맷플롯립 설정
from IPython.display import HTML
import IPython
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tf_keras
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\gram\OneDrive\FFMpeg\ffmpeg-2.1.1-win64-static\bin\ffmpeg.exe"
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_cart_pole(env,obs):
    img=env.render()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def update_scene(num, frames, patch):
    plt.close()
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, figsize=(5,6), repeat=False, interval=40):
    fig = plt.figure(figsize=figsize)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), 
                    frames=len(frames), repeat=repeat, interval=interval)
import gym

env = gym.make("CartPole-v1",render_mode="rgb_array")
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
obs = env.reset()

img = env.render()
plot_cart_pole(env,obs)

print('env.action_space :', env.action_space)
action = 1  # 오른쪽으로 가속
obs, reward, terminated, truncated, info=env.step(action)

print('obs :', obs)
print('reward :', reward)
print('terminated :', terminated)
print('truncated :', truncated)
print('info :', info)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle <0 else 1

frames, totals = [], []
for episode in range(20):
    episode_rewards = 0
    
    obs =env.reset()
    obs=obs[0]
    for step in range(1000):  # 최대 스텝을 1000번으로 설정
        img = env.render()
        frames.append(img)
        
        action = basic_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards += reward
        if terminated:
            break
    totals.append(episode_rewards)

print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))
# ! conda install -c conda-forge ffmpeg

video = plot_animation(frames, figsize=(6,4))
HTML(video.to_html5_video())  #into html5 video

# 1. layers params
n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 16  # using only 16 neurons, as it CartPole is simple env
n_outputs = 1  # probability of moving left(0)
initializer =tf.keras.initializers.he_normal
# 2. Network
tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.keras.layers.Dense(units=n_hidden, kernel_initializer=initializer,activation=tf.nn.elu)(inputs)
outputs = tf.keras.layers.Dense(units=n_outputs, activation=tf.nn.sigmoid,kernel_initializer=initializer)(hidden)
'''
Tensor("Placeholder:0", shape=(None, 4), dtype=float32) 
<Dense name=dense, built=False> 
<Dense name=dense_1, built=False>
'''


# 3. randomly select the action by output probability
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
# tf.multinomial() : output 0 by probability of outputs, and 1 by probability of 1-outputs
action = tf.random.categorical(tf.math.log(p_left_and_right), num_samples=1)

# 4. Training
n_max_steps = 1000
frames, totals = [], []

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    for episode in range(50):
        if episode % 5 == 0:
            print('episode : {}'.format(episode))
        episode_rewards = 0
        obs = env.reset()[0]
        for step in range(n_max_steps):
            img = env.render()
            frames.append(img)
            action_val = action.eval(feed_dict={inputs: obs.reshape(1, n_inputs)})
            obs, reward, terminated, truncated, info= env.step(action_val[0][0])
            episode_rewards += reward
            if terminated:
                break
        totals.append(episode_rewards)
env.close()
print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))

video = plot_animation(frames, figsize=(6,4))
HTML(video.to_html5_video())  #into html5 video

# 1. layers params
n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 16  # CartPole은 간단한 환경이므로 16개의 뉴런을 사용
n_outputs = 1  # 왼쪽(0)으로 이동할 확률을 출력
initializer = tf.keras.initializers.he_normal()

# 2. Network
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, n_outputs])
hidden = tf.keras.layers.Dense(units=n_hidden, kernel_initializer=initializer,activation=tf.nn.elu)(inputs)
logits = tf.keras.layers.Dense(units=n_outputs, activation=None,kernel_initializer=initializer)(hidden)

outputs = tf.nn.sigmoid(logits)  # 왼쪽 (0)에 대한 확률

# 3. 출력된 확률을 기반으로 랜덤하게 행동을 선택
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
# tf.multinomial() : outputs의 확률로 0, (1-outputs)의 확률로 1을 출력 
action = tf.random.categorical(tf.math.log(p_left_and_right), num_samples=1)

# 4. add loss & optimizer 
cross_entropy =tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)# 4. Training
n_max_steps = 1000

frames, totals = [], []

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    for episode in range(50):
        if episode % 5 == 0:
            print('episode : {}'.format(episode))
        episode_rewards = 0
        obs = env.reset()[0]
        for step in range(n_max_steps):
            img = env.render()
            frames.append(img)
            # angle < 0 -> proba(left)=1 else: zero
            target_probas = np.array([[1.] if obs[2] < 0 else [0.]])
            action_val, _ = sess.run([action, train_op], feed_dict={inputs: obs.reshape(1, n_inputs),labels: target_probas})
            obs, reward, terminated, truncated, info= env.step(action_val[0][0])
            episode_rewards += reward
            if terminated:
                break
        totals.append(episode_rewards)

env.close()
print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))
video = plot_animation(frames, figsize=(6,4))
HTML(video.to_html5_video())  #into html5 video
