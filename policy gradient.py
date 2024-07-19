import os, sys
import numpy as np
import tensorflow as tf
from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tf_keras
import gym

#to get some consistent value, reset the seed
def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
os.environ["TF_USE_LEGACY_KERAS"]='True'
os.environ["TF_USE_LEGACY_KERAS"]="0"

# set the matplotlib
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\gram\OneDrive\FFMpeg\ffmpeg-2.1.1-win64-static\bin\ffmpeg.exe"
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"


reset_graph()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

def update_scene(num, frames, patch):
    plt.close()
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, figsize=(5,6), repeat=False, interval=40):
    fig = plt.figure(figsize=figsize)
    print(frames)
    pre_frames=plt.imread(frames[0])
    patch = plt.imshow(pre_frames)
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), 
                    frames=len(frames), repeat=repeat, interval=interval)
import gym

n_inputs = 4
n_hidden = 16
n_outputs = 1
initializer = tf.keras.initializers.he_normal()

# Network
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.keras.layers.Dense(units=n_hidden, kernel_initializer=initializer,activation=tf.nn.elu)(inputs)
logits = tf.keras.layers.Dense(units=n_outputs, activation=None,kernel_initializer=initializer)(hidden)
outputs = tf.nn.sigmoid(logits)  # 행동 0(왼쪽)에 대한 확률
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.random.categorical(tf.math.log(p_left_and_right), num_samples=1)

# target & loss & optimizer
labels = 1. - tf.compat.v1.to_float(action)  # 타겟 확률
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, var in grads_and_vars]

gradient_placeholders, grads_and_vars_feed = [], []
for grad, var in grads_and_vars:
    gradient_placeholder = tf.compat.v1.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, var))

train_op = optimizer.apply_gradients(grads_and_vars_feed)

saver = tf.compat.v1.train.Saver()
env = gym.make('CartPole-v0')

n_iterations = 100  # 훈련 반복 횟수
n_max_steps = 1000  # 에피소드별 최대 스텝
n_games_per_update = 10  # 10번의 에피소드마다 정책을 훈련
save_iterations = 10  # 10번의 훈련 반복마다 모델을 저장
discount_rate = 0.95  # 할인 계수

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    for iteration in range(n_iterations):
        print('\r반복: {}'.format(iteration), end="")
        all_rewards, all_gradients = [], []
        for game in range(n_games_per_update):
            current_rewards, current_gradients = [], []
            obs = env.reset()[0]
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients],feed_dict={inputs: obs.reshape(1, n_inputs)})
                obs, reward, terminated, truncated, info= env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if terminated or truncated:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
            
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(train_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            tf.compat.v1.train.Saver(filename='saved_model\\my_model')

env.close()
def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()[0]
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = env.render()
            print(img)
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, terminated, truncated, info= env.step(action_val[0][0])
            if terminated or truncated:
                break
    env.close()
    return frames
frames = render_policy_net('saved_model\\my_model', action, inputs, n_max_steps=1000)
#video = plot_animation(frames, figsize=(6,4))
#HTML(video.to_html5_video())  #into html5 video
