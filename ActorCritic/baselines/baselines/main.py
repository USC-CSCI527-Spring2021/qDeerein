from train import *
import os
import DQN.Dave
import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
def playSingleEpisode(game, actor, critic, epsilon):
    episode_buffer = []
    while not game.episode_over:
        current_state = game.get_state()
        current_state_size = game.get_state_size()
        action_selected = actor.action_selection(current_state, epsilon)[0]
        action, param1, param2 = output2action(action_selected)
        new_states, rewards, episode_over, _ = game.act(action, param1, param2)
        game.update()
        reward = game.reward()
        new_state = game.get_state()
        new_state_size = game.get_state_size()

        if game.status == hfo_py.IN_GAME:
            transition = (current_state, action_selected, reward, 0, new_states)
            episode_buffer.append(transition)
    actor.replay_buffer.label(episode_buffer)

    # Store the tried experiences in the replay buffer
    actor.replay_buffer.addMultiple(episode_buffer)

    return game.total_reward, game.steps, game.status


def run(load=False, test=False):
    with tf.Session() as sess:
        print '[ddpg-Dangerous Dave] Establishing Game Environment ...'
        game = Dave()
        game.start_game()
        state_dim = game.get_state_size()

        print '[ddpg-Dangerous Dave] Creating actor / critic network ...'
        actor = ActorNetwork(sess, ACTOR_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE, SIMPLE)
        critic = CriticNetwork(sess, CRITIC_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE, actor.get_num_params(), SIMPLE)

        summary_ops, summary_vars = build_summaries()
        writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        print '[ddpg-Dangerous Dave] Initializing variables ...'
        sess.run(tf.initialize_all_variables())
        num_iter = max(actor.iter, critic.iter)

        critic.update_target_network_params()
        actor.update_target_network_params()

        saver = tf.train.Saver()
        if load:
            file_dir = [x for x in os.listdir(SUMMARY_DIR) if '.ckpt' in x]
            file_name = '%s/%s'%(SUMMARY_DIR, file_dir[-1])
            saver.restore(sess, file_name)
            print '[ddpg-Dangerous Dave] Loaded %s'%file_dir[-1]

        if load and test:
            ## Just run the pretrained model
            while True:
                if game._is_over:
                    game.reset()
                (total_reward, steps, status) = playSingleEpisode(game, actor, critic, 0.0)
                print '[ddpg-Dangerous Dave] Episode Summary -----'
                print '  total_reward:[%f]'%(total_reward)
                print '----------------------------'

        else:
            print '[ddpg-Dangerous Dave] Running iterations ...'
            while num_iter < MAX_ITER:
                if game._is_over:
                    game.reset()
                num_iter = max(actor.iter, critic.iter)

                epsilon = anneal_epsilon(num_iter)
                (total_reward, steps, status) = playSingleEpisode(game, actor, critic, epsilon)
                print '[ddpg-Dangerous Dave] Episode Summary -----'
                print '  epsilon: [%f]'%(epsilon)
                print '  total_reward:[%f]'%(total_reward)

                # Start learning once buffer has more than MEMORY_THRESHOLD examples.
                if actor.replay_buffer.size() < MEMORY_THRESHOLD:
                    print '----------------------------'
                    continue

                predicted_q_val, loss = actor_critic(actor, critic)
                qmax = np.max(predicted_q_val)
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: total_reward,
                    summary_vars[1]: qmax,
                    summary_vars[2]: loss
                })
                print '  loss: ' + str(loss)
                print '  qmax: ' + str(qmax)
                print '----------------------------'

                writer.add_summary(summary_str, num_iter)
                writer.flush()

                if num_iter % 1000 == 0:
                    save_path = saver.save(sess, "%s/model_%d.ckpt"%(SUMMARY_DIR, num_iter))


if __name__ == '__main__':
    run(load=CONTINUE, test=TEST)