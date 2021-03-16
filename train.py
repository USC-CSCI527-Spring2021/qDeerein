import numpy as np
import time
import os
import subprocess
from subprocess import Popen, PIPE
from getkeys import key_check
from ExperienceReplay import ExperienceReplay
from math import exp
from pynput.mouse import Button, Controller

num_actions = 6  # [ move_left, move_right, jump_left, jummp_right]
max_memory = 1000  # Maximum number of experiences we are storing
batch_size = 5  # Number of experiences we use for training per batch

exp_replay = ExperienceReplay(max_memory=max_memory)


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_epoch1000/Z1_model.json", "w") as json_file:  #change Z_model to model for fifa
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_epoch1000/Z1_model.h5")    #same as above
    # print("Saved model to disk")


def train(game, model, epochs,num_of_games,num_actions_t, verbose=1):
    num_actions=num_actions_t
    mouse=Controller()
    
    # We want to keep track of the progress of the AI over time, so we save its win count history
    loss_hist = []
    average_points=[]
    current_step=0

    for ga in range(num_of_games):
        time.sleep(1)
        reward_count=0
        subprocess.Popen(".\\dave.exe")
        game.start_game()
        game.reset()
        game_over = False
        
        print('will begin in 5 seconds')
        time.sleep(5)
        loss=0.0
        average_value=0
        

        for e in range(1,epochs+1):
            current_step+=1       
            epsilon = 0.01 + 0.99*exp(-1.*current_step*0.001)
            input_t = game.observe()
            if e==1:
                time.sleep(1)
            input_tm1 = input_t

            n=np.random.rand()
            if n <= epsilon:
                action = int(np.random.randint(0, num_actions, size=1))
                print('action is: ',action)
            else:
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                print("Q value ",q)
                action = np.argmax(q[0])
                print('action is: ',action)

            # apply action, get rewards and new state
            input_t, reward, game_over = game.act(action)
            print("ingame_reward",reward)
            if reward==0.2 or reward==1:
                reward_count+=1
                average_value+=reward_count/e
                
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)
            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
        
            # train model on experiences
            batch_loss = model.train_on_batch(inputs, targets)
            print('batch loss',batch_loss,targets)
            loss += batch_loss
            # menu control
            keys = key_check()
            if 'O' in keys:
                print('Quitting!')
                return
            print("Step {:03d}/{:03d}".format(e,epochs))
            if game_over:
                print('game over! restarting')
                mouse.position=(1280,280)
                mouse.click(Button.left,1)
                break
                

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f}".format(ga, num_of_games, loss))
            #print('Epoch: {}, Loss: {}, Accuracy: {}'.format(e+1,loss,win_cnt/e*100))
        
        loss_hist.append(loss)
        average_points.append(average_value)
        mouse.position=(1280,280)
        mouse.click(Button.left,1)
        game.exit_game()

    save_model(model)
    return loss_hist,average_points
