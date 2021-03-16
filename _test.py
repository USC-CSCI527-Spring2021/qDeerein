import time
import numpy as np
from getkeys import key_check
import subprocess

num_actions = 5
def _test(game, model, n_games,num_of_games,num_actions_t, verbose=1):
    num_actions = num_actions_t
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    # Epochs is the number of games we play
    for e in range(n_games):
        # Resetting the game
        time.sleep(1)
        subprocess.Popen(".\\dave.exe")
        game.start_game()
        game.reset()
        game_over = False
        input_t = game.observe()
        paused = False
        while not game_over:
            if not paused:
                input_tm1 = input_t
                # Choose yourself
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                print('q values=' + str(q[0]))
                q_dict = dict()
                for i in range(len(q[0])):
                    q_dict[q[0][i]] = i

                q_value = sorted(q[0], reverse=True)
                max_iteration = 0
                # action = np.argmax(q[0])
                action = q_dict[q_value[max_iteration]]

                # apply action, get rewards and new state
                input_t, reward, game_over = game.act(action)
                # If we managed to catch the fruit we add 1 to our win counter
                # while input_t == input_tm1:
                while reward <= 0 and max_iteration < 6:
                    # q = model.predict(input_tm1)
                    # We pick the action with the highest expected reward
                    # print("Whole q is: "+ str(q))
                    print('q values=' + str(q[0]))
                    # action = np.argmax(q[0])
                    max_iteration += 1
                    action = q_dict[q_value[max_iteration]]
                    # apply action, get rewards and new state
                    input_t, reward, game_over = game.act(action)
                    
                if reward == 1:
                    win_cnt += 1
            # menu control
            keys = key_check()
            if 'P' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)
            elif 'O' in keys:
                print('Quitting!')
                return

        if verbose > 0:
            print("Game {:03d}/{:03d} | Win count {}".format(e, n_games, win_cnt))
        win_hist.append(win_cnt)
    return win_hist
