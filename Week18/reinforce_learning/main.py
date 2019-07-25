import numpy as np
import pickle
import sys

def preprocess(img):
    """Preprocess images with RGB into 1D float vector
    :param img: [210, 160, 3], uint8
    :return out: [6400], float
    """
    # crop, [210, 160, 3] -> [160, 160, 3]
    img = img[35:195]
    # downsample by factor of 2, using only one channel, [160, 160]
    img = img[::2, ::2, 0]
    # erase background
    img[img == 144] = 0
    img[img == 109] = 0
    # everything else (paddles, ball) just set to 1
    img[img != 0] = 1
    out = img.astype(np.float).ravel()
    return out

def sigmoid(x):
    # sigmoid "squashing" function to interval [0,1]
    return 1.0 / (1.0 + np.exp(-x))

def policy_forward(model, x):
    """Forward, return probability of taking action 2 and hidden state
    """
    # [hidden_dim]
    hidden = np.dot(model["W1"], x)
    # [hidden_dim]
    hidden[hidden < 0] = 0 # ReLU nonlinearity
    # scalar
    logp = np.dot(model["W2"], hidden)
    prob = sigmoid(logp)
    return prob, hidden

def policy_backward(episode_hiddens, episode_dlogps, model, episode_x):
    """Backward padd
    :param episode_hiddens: array of intermediate hidden states,
                            [time, hidden_dim]
    :param episode_dlogps: array of intermediate logp, [time]
    :param episode_x: array of x
    """
    # [hidden_dim] ??
    dW2 = np.dot(episode_hiddens.T, episode_dlogps).ravel()
    dhidden = np.outer(episode_dlogps, model["W2"])
    dW1 = np.dot(dhidden.T, episode_x)
    return {"W1": dW1, "W2": dW2}

def discount_rewards(rewards, gamma):
    """Task 1D float array of rewards and compute discounted reward
    :param rewards: 1D float array of rewards
    :return out: discount_rewards
    """
    out = np.zeros_like(rewards)
    running_add = 0

    for time in reversed(range(0, rewards.size)):
        # reset the sum, since this was a game boundary (pong specific!)
        if rewards[time] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[time]
        out[time] = running_add
    return out


def train(model, env, input_dim, batch_size, alpha, gamma,
          learning_rate, render=False):
    observation = env.reset()
    prev_x = None
    xs, hiddens, dlogps, drs = [], [], [], []
    reward_sum = 0
    episode_number = 0
    grad_buffer = {key: np.zeros_like(value) for key, value in model.items()}
    rmsprop_cache = {key: np.zeros_like(value) for key, value in model.items()}
    running_reward = None


    while True:
        if render == True:
            env.render()

        # Preprocess the observation, set input to network
        # to be difference image
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x

        # forward the policy network and sample an action from the
        # returned probability
        prob, hidden = policy_forward(model, x)
        if np.random.uniform() < prob:
            action = 2
        else:
            action = 3

        # record various intermediates (need later for backprop)
        xs.append(x)
        hiddens.append(hidden)

        # fake label by action
        if action == 2:
            y = 1
        else:
            y = 0

        dlogps.append(y - prob)

        # step the environment and get new measurement
        observation, reward, done, info = env.step(action)

        # record reward ( has to be done after we call step() to get reward
        # for previous action)
        reward_sum += reward
        drs.append(reward)

        # an episode finished
        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients
            # and reward for this episode
            episode_x = np.vstack(xs)
            episode_hiddens = np.vstack(hiddens)
            episode_dlogps = np.vstack(dlogps)
            episode_reward = np.vstack(drs)
            # reset array memory
            xs, hiddens, dlogps, drs = [], [], [], []

            # compute the discounted reward backwards through time
            discounted_episode_reward = discount_rewards(episode_reward,
                                                         gamma)
            # standardlize the rewards to be unit normal (helps control the
            # gradient estimator variance)
            discounted_episode_reward -= np.mean(discounted_episode_reward)
            discounted_episode_reward /= np.std(discounted_episode_reward)

            # Modulate gradient with advantage (PG magic happens right here)
            episode_dlogps *= discounted_episode_reward
            grad = policy_backward(episode_hiddens, episode_dlogps, model,
                                   episode_x)

            # accumalate grad over batch
            for key in model:
                grad_buffer[key] += grad[key]

            if episode_number % batch_size == 0:
                for key, value in model.items():
                    grad = grad_buffer[key]
                    rmsprop_cache[key] = alpha * rmsprop_cache[key] + \
                                         (1 - alpha) * grad * 2
                    model[key] += learning_rate * grad / \
                                  np.sqrt(rmsprop_cache[key] + 1e-5)
                    # reset batch gradient buffer
                    grad_buffer[key] = np.zeros_like(value)

            # boring booking
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum + 0.01

            print('resetting env. episode reward total was %f. \
                   running mean: %f' % (reward_sum, running_reward))

            if episode_number % 100 == 0:
                with open("save.p", "wb") as f:
                    pickle.dump(model, f)
            reward_sum = 0
            observation = env.reset()
            prev_x = None

        # Pong has either +1 or -1 reward exactly when game ends.
        if reward != 0:
            print('ep %d: game finished, reward: %f' % \
                  (episode_number, reward) + \
                  ('' if reward == -1 else ' !!!!!!!!'))
