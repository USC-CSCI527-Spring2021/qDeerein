import numpy as np
import pytesseract as pt
import cv2
import win32gui
import mss
import mss.tools
from CNN import CNN
from PIL import Image
from PIL import ImageGrab
from pynput.mouse import Button, Controller
from grabscreen import grab_screen
from directkeys import *
import time
from grabImageUsingCV2 import fetchScoreFromImage

mouse=Controller()
class Dave(object):
    cnn_graph = CNN()
    def __init__(self):
        self.rward=0
        # self.reset()
        self.visit=0
        self.previous_screen=[]
        self.previous_compare_screen=[]
        # self.previous_gold_screen=[]
        self.previous_screen_score = 0
        self.curr_score = 0
        self.prev_unprocessed = 0
        self.flagTrophy = False

    def _get_reward(self, action):
        flag1=False
        flag2=False
        flag3=False
        flagTrophy = False
        screen = mss.mss().grab((0,0,1366,768))
        mss.tools.to_png(screen.rgb,screen.size,output='screen.jpg')
        self.visit+=1
        screen = np.asarray(screen)
        if self.visit > 1 :
            self.curr_score = 0
            reward_scrn = mss.mss().grab((310,0,470,47))
            mss.tools.to_png(reward_scrn.rgb,reward_scrn.size,output='reward.jpg')
            # reward_scrn = Image.open("reward.jpg")
            reward_scrn = cv2.imread("reward.jpg")
            self.curr_score = fetchScoreFromImage(reward_scrn) 
            prev_score = self.previous_screen_score
            if self.curr_score != self.prev_unprocessed:
                self.prev_unprocessed = self.curr_score
                if self.curr_score < self.previous_screen_score:
                    self.curr_score = self.previous_screen_score + 100
                elif(self.curr_score - self.previous_screen_score >= 1000):
                    self.flagTrophy = True

                diff_in_score = self.curr_score - self.previous_screen_score
                if(diff_in_score >= 50):
                    flag1 = True
                self.previous_screen_score = self.curr_score
            else:
                self.curr_score = self.previous_screen_score
            print("\nPrevious Score was %d and Current score is %d \n" % (prev_score, self.curr_score))

        compare_screen=screen[50:,0:]
        if self.visit>1:
            diff2=cv2.subtract(compare_screen,self.previous_compare_screen)
            b2,g2,r2,a2=cv2.split(diff2)
            #print(cv2.countNonZero(b1) , cv2.countNonZero(g1) , cv2.countNonZero(r1))
            max2=max(cv2.countNonZero(b2) , cv2.countNonZero(g2) , cv2.countNonZero(r2))
            print("difference",max2)
            if max2<1300:
                flag3=True  
        self.previous_compare_screen=compare_screen                
        
        # lives_screen= screen[55:75,70:150]
        # if self.visit>1:
        #     diff1=cv2.subtract(lives_screen,self.previous_screen)
        #     b1,g1,r1=cv2.split(diff1)          
        #     if cv2.countNonZero(b1) > 0 or cv2.countNonZero(g1) > 0 or cv2.countNonZero(r1) > 0:
        #         flag2=True                
        # self.previous_screen=lives_screen
        
        # flag 2 is for lives
        # if flag2:
        #     return -1
        # flag 1 is true if any gem is collected
        if flagTrophy:
            return 1
        if flag1:
            return 0.2
        # flag 3 is true if there is no change in new frame.
        if flag3:
            return -0.1
        ingame_reward = 0.01
        return ingame_reward

    def _is_over(self, reward):
        if self.curr_score == 2200:
            is_over = True
        else:
            is_over = False
        return is_over
    
    
    def observe(self):
        screen = np.asarray(ImageGrab.grab(bbox=(0,50,1336,668)))  
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        keys_to_press = [[uparrow,leftarrow],[leftarrow],[uparrow,leftarrow],[uparrow,rightarrow],[rightarrow],[uparrow,rightarrow]]
        PressKey(keys_to_press[action][0])
        time.sleep(0.3)
        ReleaseKey(keys_to_press[action][0])
        i=1
        if(len(keys_to_press[action]) >=1):
            while i<len(keys_to_press[action]):
                PressKey(keys_to_press[action][i])
                time.sleep(0.4)
                ReleaseKey(keys_to_press[action][i])
                i+=1
        time.sleep(0.6)
        reward = self._get_reward(action)
        game_over = self._is_over(reward)
        return self.observe(), reward, game_over

    def reset(self):
        self.rward=0
        self.visit=0
        self.previous_screen=[]
        self.previous_compare_screen=[]
        self.previous_screen_score = 0
        self.curr_score = 0
        self.prev_unprocessed = 0
        self.flagTrophy = False

    def start_game(self):
        PressKey(0x39)
        time.sleep(1)
        ReleaseKey(0x39)
        time.sleep(1)

    def exit_game(self):
        PressKey(0x01)
        time.sleep(1)
        ReleaseKey(0x01)
        time.sleep(1)
        PressKey(0x15)
        time.sleep(1)
        ReleaseKey(0x15)
        time.sleep(1)

