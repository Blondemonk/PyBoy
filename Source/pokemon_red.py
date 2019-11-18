#! /usr/local/bin/python2
# -*- encoding: utf-8 -*-
#
# Authors: Asger Anders Lund Hansen, Mads Ynddal and Troels Ynddal
# License: See LICENSE file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import traceback
import time
import os.path
import os
import sys
import platform
from PyBoy.Logger import logger

if platform.system() != "Windows":
    from Debug import Debug
from PyBoy import PyBoy

from PyBoy.GameWindow import SdlGameWindow as Window

# from agents import PokemonAgent as Agent
from rl.agents.dqn import DQNAgent as Agent

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

# from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback

from PyBoy.WindowEvent import WindowEvent

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        # return np.clip(reward, -1., 1.)
        return reward

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
nb_actions = 6
actions_list = [
    WindowEvent.PressArrowUp,
    WindowEvent.PressArrowDown,
    WindowEvent.PressArrowRight,
    WindowEvent.PressArrowLeft,
    WindowEvent.PressButtonA,
    WindowEvent.PressButtonB,
    WindowEvent.PressButtonSelect,
    WindowEvent.PressButtonStart,
]

model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
    nb_steps=1000000)
dqn = Agent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
            processor=processor, nb_steps_warmup=10000, gamma=.99, target_model_update=10000,
            train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
dqn.load_weights(weights_filename)


if __name__ == "__main__":
    # Automatically bump to '-OO' optimizations
    if __debug__:
        os.execl(sys.executable, sys.executable, '-OO', *sys.argv)

    bootROM = "ROMs/DMG_ROM.bin"
    scale = 1
    debug = "debug" in sys.argv and platform.system() != "Windows"

    # Verify directories
    if not bootROM is None and not os.path.exists(bootROM):
        print ("Boot-ROM not found. Please copy the Boot-ROM to '%s'. Using replacement in the meanwhile..." % bootROM)
        bootROM = None

    try:
        filename = "../ROMs/Pokemon Red.gb"

        # Start PyBoy and run loop
        pyboy = PyBoy(Window(scale=scale), filename, bootROM)
        step = 0
        while not pyboy.tick():
            try:
                # ((160,144) * scale)-sized black/white array
                screen_array = pyboy.getScreenBuffer().getScreenBuffer()
                # print screen_array.shape
                observation = dqn.processor.process_observation(screen_array)
                action = dqn.forward(observation)
                pyboy.sendInput(actions[action])
            except Exception as e:
                print e
            pass
        pyboy.stop()

    except KeyboardInterrupt:
        print ("Interrupted by keyboard")
        pyboy.stop()
    except Exception as ex:
        traceback.print_exc()
