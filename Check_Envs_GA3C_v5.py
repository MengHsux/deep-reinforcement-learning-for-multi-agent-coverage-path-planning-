# NEW COLORS 108.04.24
# output=gray colors
import numpy as np
import pygame
import time

# Define some colors
COLORS = 3
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
BLUE = np.array((60, 150, 255))
PURPLE = np.array((153, 47, 185))
RED_PROBE = np.array((230, 90, 80))
YELLOW = np.array((235, 226, 80))
HUR = np.array((100, 150, 255))
BLUES = np.array((160, 90, 80))

BACKGROUND_COLORS = 255
BUFFER_COLORS = 170
PROBE_COLORS = 220
OTHER_COLORS = 129

NUM_COLORS = []
for num in range(COLORS):
    NUM_COLORS.append(int(OTHER_COLORS * (1 - num / COLORS)))

print(NUM_COLORS)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 1
HEIGHT = 1
WIDTH_sc = 15
HEIGHT_sc = 15

# This sets the margin between each cell
MARGIN = 0
MARGIN_sc = 2

# Probe's location when the environment initialize
Initial = [(2, 2), (4, 4), (2, 4), (4, 2), (6, 4), (4, 6), (6, 6), (5, 5)]


class wafer_check():
    def __init__(self, wafer, probe, probe1, location=(0, 0), mode=0, training_time=60, training_steps=0):
        self._envs = np.array(wafer)
        self._envs_nan = np.zeros(self._envs.shape)

        self._probe = np.array(probe, np.int)
        self._probe1 = np.array(probe1, np.int)
        self._probe2 = np.array(probe1, np.int)

        self.envsY, self.envsX = self._envs.shape
        self.wafer_len = self.envsY * self.envsX

        self.probY, self.probX = self._probe.shape
        self.probZ = max(self.probY, self.probX)
        self.probY1, self.probX1 = self._probe1.shape
        self.probZ1 = max(self.probY1, self.probX1)
        self.probY2, self.probX2 = self._probe2.shape
        self.probZ2 = max(self.probY2, self.probX2)

        self.envs_list = [(b, a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b, a] == -1]
        self.envs_len = len(self.envs_list)
        self.envs_list1 = [(b, a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b, a] == -1]
        self.envs_len1 = len(self.envs_list1)

        self.probe_list = [(b, a) for b in range(self.probY) for a in range(self.probX) if self._probe[b, a] == 1]
        self.probe_len = len(self.probe_list)
        self.probe_list1 = [(b, a) for b in range(self.probY1) for a in range(self.probX1) if self._probe1[b, a] == 1]
        self.probe_len1 = len(self.probe_list1)
        self.probe_list2 = [(b, a) for b in range(self.probY2) for a in range(self.probX2) if self._probe2[b, a] == 1]
        self.probe_len2 = len(self.probe_list2)

        self.size = [(self.envsX * WIDTH + (self.envsX + 1) * MARGIN),
                     (self.envsY * HEIGHT + (self.envsY + 1) * MARGIN)]
        self.size1 = [(self.envsX * WIDTH_sc + (self.envsX + 1) * MARGIN_sc),
                      (self.envsY * HEIGHT_sc + (self.envsY + 1) * MARGIN_sc)]
        self._output = np.full((self.size[1], self.size[0]), BACKGROUND_COLORS, np.int)

        self.location = np.array(Initial)
        self.location1 = np.array(Initial)
        self.location2 = np.array(Initial)

        self.action_space = ['None', 'Down', 'Right', 'Up', 'Left', 'Down-Right', 'Up-Right', 'Up-Left', 'Down-Left']
        self.action_space_num = int((len(self.action_space) - 1) * 1)  # self.probZ)
        self.available = np.zeros(self.action_space_num, dtype=np.float32)
        self.num_max = COLORS

        self.reward_value = 0
        self.reward_value1 = 0
        self.reward_value2 = 0

        self.envs_mean = None
        self.envs_std = None
        self.mode = mode

        if training_time > 0:
            self.training_time = training_time
        else:
            self.training_time = np.inf

        if training_steps > 0:
            self.training_steps = training_steps
        else:
            self.training_steps = np.inf

        if self.mode == 1:
            self.sc = pygame.display.set_mode(self.size1)

        self.reset_observation()
        self.reset()

    @staticmethod
    def rect(column, row):
        rect = [(MARGIN_sc + WIDTH_sc) * column + MARGIN_sc,
                (MARGIN_sc + HEIGHT_sc) * row + MARGIN_sc,
                WIDTH_sc,
                HEIGHT_sc]
        return rect

    @staticmethod
    def draw_plt(output, y, x, color):  # X : column, Y : row
        for h in range(HEIGHT):
            for w in range(WIDTH):
                output_h = y * HEIGHT + h
                output_w = x * WIDTH + w
                output[output_h][output_w] = color

    def reset(self):
        # reset the environment
        self.y, self.x = self.location[np.random.randint(len(self.location))]
        self.y1, self.x1 = self.location1[np.random.randint(len(self.location1))]
        self.y2, self.x2 = self.location2[np.random.randint(len(self.location2))]

        self.y_last, self.x_last = self.y, self.x
        self.y_last1, self.x_last1 = self.y1, self.x1
        self.y_last2, self.x_last2 = self.y2, self.x2

        self.steps = 0
        self.dist = 0
        self.dist1 = 0
        self.dist2 = 0

        self.num_color = np.zeros(self.num_max + 2, np.int)
        self.num_color1 = np.zeros(self.num_max + 2, np.int)
        self.num_color2 = np.zeros(self.num_max + 2, np.int)

        self.action = 'None'
        self.action1 = 'None'
        self.action2 = 'None'

        self.reward_value = 0
        self.reward_value1 = 0
        self.reward_value2 = 0

        self.envs = np.copy(self._envs_nan)
        self.output = np.copy(self._output)

        if self.mode == 1:
            self.reset_envs()

        for b in range(self.probY):
            for a in range(self.probX):
                if self._probe[b][a] == 1 and not np.isnan(self.envs[self.y + b][self.x + a]):  # fix 0505
                    self.envs[self.y + b][self.x + a] = 1

        for b in range(self.probY1):
            for a in range(self.probX1):
                if self._probe1[b][a] == 1 and not np.isnan(self.envs[self.y1 + b][self.x1 + a]):  # fix 0505
                    self.envs[self.y1 + b][self.x1 + a] = 1

        for b in range(self.probY2):
            for a in range(self.probX2):
                if self._probe2[b][a] == 1 and not np.isnan(self.envs[self.y2 + b][self.x2 + a]):  # fix 0505
                    self.envs[self.y2 + b][self.x2 + a] += 1

        self.num_color_last = np.zeros(self.num_max + 2, np.int)
        self.num_color_last[-1] = self.envs_len
        self.num_color_last[0] = (self._envs == 0).sum()

        self.num_color_last1 = np.zeros(self.num_max + 2, np.int)
        self.num_color_last1[-1] = self.envs_len1
        self.num_color_last1[0] = (self._envs == 0).sum()

        self.num_color_last2 = np.zeros(self.num_max + 2, np.int)
        self.num_color_last2[-1] = self.envs_len1
        self.num_color_last2[0] = (self._envs == 0).sum()

        self.time_end = time.time() + self.training_time

        self.step()
        return self.output, self.available

    def step(self, action=None, action1=None, action2=None):
        # Agent's action
        now = time.time()

        if action != None:
            act = ((action) % 8)
            step = int((action) / 8) + 1

        if action1 != None:
            act1 = ((action1) % 8)
            step1 = int((action1) / 8) + 1

        if action2 != None:
            act2 = ((action2) % 8)
            step2 = int((action2) / 8) + 1

        self.done = 0
        self.envs_mean = None
        self.envs_std = None
        self.time_is_end = 0
        self.steps_is_end = 0
        self.episode_is_end = 0
        self.reward_value = 0
        self.reward_value1 = 0
        self.reward_value2 = 0

        if now < self.time_end and self.steps < self.training_steps:

            y = self.y
            x = self.x
            y_diff = self.envsY - self.probY
            x_diff = self.envsX - self.probX
            l = 0
            probe_list = self.probe_list
            invalid = 0
            self.steps += 1

            # move the probe
            if action == None:
                invalid = -1
                self.steps -= 1
                self.action = 'None'
            elif step > self.probZ:
                invalid = -1
                self.steps -= 1
                self.action = 'None'
            elif act == 0:
                if (y + step - 1) < y_diff:
                    y += step
                    invalid = 0
                    self.action = 'Down'
                else:
                    invalid = 1
            elif act == 1:
                if (x + step - 1) < x_diff:
                    x += step
                    invalid = 0
                    self.action = 'Right'
                else:
                    invalid = 1
            elif act == 2:
                if (y - step + 1) > 0:
                    y -= step
                    invalid = 0
                    self.action = 'Up'
                else:
                    invalid = 1
            elif act == 3:
                if (x - step + 1) > 0:
                    x -= step
                    invalid = 0
                    self.action = 'Left'
                else:
                    invalid = 1
            elif act == 4:
                if (y + step - 1) < y_diff and (x + step - 1) < x_diff:
                    y += step
                    x += step
                    invalid = 0
                    self.action = 'Down-Right'
                else:
                    invalid = 1
            elif act == 5:
                if (y - step + 1) > 0 and (x + step - 1) < x_diff:
                    y -= step
                    x += step
                    invalid = 0
                    self.action = 'Up-Right'
                else:
                    invalid = 1
            elif act == 6:
                if (y - step + 1) > 0 and (x - step + 1) > 0:
                    y -= step
                    x -= step
                    invalid = 0
                    self.action = 'Up-Left'
                else:
                    invalid = 1
            elif act == 7:
                if (y + step - 1) < y_diff and (x - step + 1) > 0:
                    y += step
                    x -= step
                    invalid = 0
                    self.action = 'Down-Left'
                else:
                    invalid = 1
            else:
                invalid = -1
                self.action = 'None'

            if invalid == 1:

                self.action = 'Invalid'

            elif invalid == -1:
                y = self.y
                x = self.x

            elif invalid == 0:
                self.y = y
                self.x = x

                for c in range(len(probe_list)):
                    self.envs[y + probe_list[c][0]][x + probe_list[c][1]] += 1

        elif now >= self.time_end:
            invalid = -1
            self.time_is_end = 1

        if self.steps >= self.training_steps:
            invalid = -1
            self.steps_is_end = 1

        self.check()

        self.observation()

        self.action_available()

        now = time.time()

        if now < self.time_end and self.steps < self.training_steps:

            y1 = self.y1
            x1 = self.x1
            y_diff1 = self.envsY - self.probY1
            x_diff1 = self.envsX - self.probX1

            l = 0

            probe_list1 = self.probe_list1

            invalid1 = 0
            self.steps += 1

            # move the probe2
            if action1 == None:
                invalid1 = -1
                self.steps -= 1
                self.action1 = 'None'
            elif step1 > self.probZ1:
                invalid1 = -1
                self.steps -= 1
                self.action1 = 'None'
            elif act1 == 0:
                if (y1 + step1 - 1) < y_diff1:
                    y1 += step1
                    invalid1 = 0
                    self.action1 = 'Down'
                else:
                    invalid1 = 1
            elif act1 == 1:
                if (x1 + step1 - 1) < x_diff1:
                    x1 += step1
                    invalid1 = 0
                    self.action1 = 'Right'
                else:
                    invalid1 = 1
            elif act1 == 2:
                if (y1 - step1 + 1) > 0:
                    y1 -= step1
                    invalid1 = 0
                    self.action1 = 'Up'
                else:
                    invalid1 = 1
            elif act1 == 3:
                if (x1 - step1 + 1) > 0:
                    x1 -= step1
                    invalid1 = 0
                    self.action1 = 'Left'
                else:
                    invalid1 = 1
            elif act1 == 4:
                if (y1 + step1 - 1) < y_diff1 and (x1 + step1 - 1) < x_diff1:
                    y1 += step1
                    x1 += step1
                    invalid1 = 0
                    self.action1 = 'Down-Right'
                else:
                    invalid1 = 1
            elif act1 == 5:
                if (y1 - step1 + 1) > 0 and (x1 + step1 - 1) < x_diff1:
                    y1 -= step1
                    x1 += step1
                    invalid1 = 0
                    self.action1 = 'Up-Right'
                else:
                    invalid1 = 1
            elif act1 == 6:
                if (y1 - step1 + 1) > 0 and (x1 - step1 + 1) > 0:
                    y1 -= step1
                    x1 -= step1
                    invalid1 = 0
                    self.action1 = 'Up-Left'
                else:
                    invalid1 = 1
            elif act1 == 7:
                if (y1 + step1 - 1) < y_diff1 and (x1 - step1 + 1) > 0:
                    y1 += step1
                    x1 -= step1
                    invalid1 = 0
                    self.action1 = 'Down-Left'
                else:
                    invalid1 = 1
            else:
                invalid1 = -1
                self.action1 = 'None'

            if invalid1 == 1:

                self.action1 = 'Invalid'

            elif invalid1 == -1:
                y1 = self.y1
                x1 = self.x1

            elif invalid1 == 0:
                self.y1 = y1
                self.x1 = x1

                for c in range(len(probe_list1)):
                    self.envs[y1 + probe_list1[c][0]][x1 + probe_list1[c][1]] += 1

        elif now >= self.time_end:
            invalid1 = -1
            self.time_is_end = 1

        if self.steps >= self.training_steps:
            invalid1 = -1
            self.steps_is_end = 1

        self.check1()

        self.observation1()

        self.action_available1()

        now = time.time()

        if now < self.time_end and self.steps < self.training_steps:
            y2 = self.y2
            x2 = self.x2
            m2 = self.envsY
            n2 = self.envsX
            i2 = self.probY2
            j2 = self.probX2
            k2 = self.probZ2

            probe_list2 = self.probe_list2
            probe_len2 = self.probe_len2

            invalid2 = 0
            self.steps += 1

            # move the probe2
            if action2 == None:
                invalid2 = -1
                self.steps -= 1
                self.action2 = 'None'
            elif step2 > k2:
                invalid2 = -1
                self.steps -= 1
                self.action2 = 'None'
            elif act2 == 0:
                if (y2 + step2 - 1) < (m2 - i2):
                    y2 += step2
                    invalid2 = 0
                    self.action2 = 'Down'
                else:
                    invalid2 = 1
            elif act2 == 1:
                if (x2 + step2 - 1) < (n2 - j2):
                    x2 += step2
                    invalid2 = 0
                    self.action2 = 'Right'
                else:
                    invalid2 = 1
            elif act2 == 2:
                if (y2 - step2 + 1) > 0:
                    y2 -= step2
                    invalid2 = 0
                    self.action2 = 'Up'
                else:
                    invalid2 = 1
            elif act2 == 3:
                if (x2 - step2 + 1) > 0:
                    x2 -= step2
                    invalid2 = 0
                    self.action2 = 'Left'
                else:
                    invalid2 = 1
            elif act2 == 4:
                if (y2 + step2 - 1) < (m2 - i2) and (x2 + step2 - 1) < (n2 - j2):
                    y2 += step2
                    x2 += step2
                    invalid2 = 0
                    self.action2 = 'Down-Right'
                else:
                    invalid2 = 1
            elif act2 == 5:
                if (y2 - step2 + 1) > 0 and (x2 + step2 - 1) < (n2 - j2):
                    y2 -= step2
                    x2 += step2
                    invalid2 = 0
                    self.action2 = 'Up-Right'
                else:
                    invalid2 = 1
            elif act2 == 6:
                if (y2 - step2 + 1) > 0 and (x2 - step2 + 1) > 0:
                    y2 -= step2
                    x2 -= step2
                    invalid2 = 0
                    self.action2 = 'Up-Left'
                else:
                    invalid2 = 1
            elif act2 == 7:
                if (y2 + step2 - 1) < (m2 - i2) and (x2 - step2 + 1) > 0:
                    y2 += step2
                    x2 -= step2
                    invalid2 = 0
                    self.action2 = 'Down-Left'
                else:
                    invalid2 = 1
            else:
                invalid2 = -1
                self.action2 = 'None'

            for c in range(probe_len2):
                if invalid2 == 0:
                    if self._envs[y2 + probe_list2[c][0]][x2 + probe_list2[c][1]] == -1:
                        invalid2 = 1
                        y2 = self.y2
                        x2 = self.x2
                        break
                else:
                    break

            if invalid2 == 1:
                y2 = self.y2
                x2 = self.x2

                self.action2 = 'Invalid'

            elif invalid2 == -1:
                y2 = self.y2
                x2 = self.x2

            elif invalid2 == 0:

                self.y2 = y2
                self.x2 = x2

                for c in range(len(probe_list2)):
                    self.envs[y2 + probe_list2[c][0]][x2 + probe_list2[c][1]] += 1

        elif now >= self.time_end:
            invalid2 = -1
            self.time_is_end = 1

        elif self.steps >= self.training_steps:
            invalid2 = -1
            self.steps_is_end = 1

        self.check2()

        self.observation2()

        self.action_available2()

        if self.mode == 1:
            self.build_envs()
            time.sleep(0.01)

        self.y_last = self.y
        self.x_last = self.x

        self.y_last1 = self.y1
        self.x_last1 = self.x1

        self.y_last2 = self.y2
        self.x_last2 = self.x2

        if self.steps_is_end == 1:
            self.steps = 0

        if self.time_is_end == 1:
            self.steps = 0
            self.time_end = time.time() + self.training_time

        return self.output, self.reward_value, self.reward_value1, self.reward_value2, self.done, self.available, self.envs_mean, self.envs_std

    def check(self):
        # calculate the reward
        self.num_color[-1] = self.envs_len
        for n in range(0, self.num_max):
            self.num_color[n] = (self.envs == n).sum()

        self.num_color[-2] = self.wafer_len - sum(self.num_color) + self.num_color[-2]  # >num_max

        self.dist = np.sqrt(np.square(self.y - self.y_last) + np.square(self.x - self.x_last))

        if self.action != "None":

            # 1st reward
            if self.num_color_last[0] - self.num_color[0] > 0:
                self.reward_value += ((self.num_color_last[0] - self.num_color[0]) * 0.01)
                if self.num_color_last[0] - self.num_color[0] == self.probe_len:
                    self.reward_value += ((self.num_color_last[0] - self.num_color[0]) * 0.01)

            # 2nd reward
            for num in range(2, self.num_max + 1):
                if self.num_color[num] - self.num_color_last[num] > 0:
                    self.reward_value -= (((self.num_color[num] - self.num_color_last[num]) * num) * 0.003)

            # 3rd reward
            if np.array_equal(self.num_color, self.num_color_last):
                self.reward_value -= 0.1

            # 4th reward
            self.reward_value -= self.dist * 0.01

        # Initialize the environment
        if self.num_color[0] == 0 or self.time_is_end == 1 or self.steps_is_end == 1:
            self.envs_mean = np.nanmean(self.envs)
            self.envs_std = np.nanstd(self.envs)

            # Stop the screen when the episode is end.
            if self.mode == 1:
                self.build_envs()
                time.sleep(0.1)

            # Initialize the environment
            self.action = 'None'
            self.done = 1
            self.y, self.x = self.location[np.random.randint(len(self.location))]
            self.y_last, self.x_last = self.y, self.x
            self.dist = 0

            self.num_color = np.zeros(self.num_max + 2, np.int)
            self.envs = np.copy(self._envs_nan)
            self.output = np.copy(self._output)
            if self.mode == 1:
                self.reset_envs()

            for b in range(self.probY):
                for a in range(self.probX):
                    if self._probe[b][a] == 1:
                        self.envs[self.y + b][self.x + a] = 1

            self.envs_show = np.copy(self.envs)
            self.num_color[-1] = self.envs_len
            self.num_color[0] = (self.envs == 0).sum()
            self.num_color[1] = (self.envs == 1).sum()

            if self.time_is_end != 1 and self.steps_is_end != 1:
                self.episode_is_end = 1

                # 5th reward
                self.reward_value += 1
                self.steps = 0

        self.num_color_last = np.copy(self.num_color)

    def check1(self):
        # calculate the reward
        self.num_color1[-1] = self.envs_len1
        for n in range(0, self.num_max):
            self.num_color1[n] = (self.envs == n).sum()

        self.num_color1[-2] = self.wafer_len - sum(self.num_color1) + self.num_color1[-2]  # >num_max

        self.dist1 = np.sqrt(np.square(self.y1 - self.y_last1) + np.square(self.x1 - self.x_last1))

        if self.action1 != "None":

            # 1st reward
            if self.num_color_last1[0] - self.num_color1[0] > 0:
                self.reward_value1 += ((self.num_color_last1[0] - self.num_color1[0]) * 0.01)
                if self.num_color_last1[0] - self.num_color1[0] == self.probe_len1:
                    self.reward_value1 += ((self.num_color_last1[0] - self.num_color1[0]) * 0.01)

            # 2nd reward
            for num in range(2, self.num_max + 1):
                if self.num_color1[num] - self.num_color_last1[num] > 0:
                    self.reward_value1 -= (((self.num_color1[num] - self.num_color_last1[num]) * num) * 0.003)

            # 3rd reward
            if np.array_equal(self.num_color1, self.num_color_last1):
                self.reward_value1 -= 0.1

            # 4th reward
            self.reward_value1 -= self.dist1 * 0.01

        # Initialize the environment
        if self.num_color1[0] == 0 or self.time_is_end == 1 or self.steps_is_end == 1:
            self.envs_mean = np.nanmean(self.envs)
            self.envs_std = np.nanstd(self.envs)

            # Stop the screen when the episode is end.
            if self.mode == 1:
                self.build_envs()
                time.sleep(0.1)

            # Initialize the environment
            self.action1 = 'None'
            self.done = 1

            self.y1, self.x1 = self.location1[np.random.randint(len(self.location1))]
            self.y_last1, self.x_last1 = self.y1, self.x1
            self.dist1 = 0

            self.num_color1 = np.zeros(self.num_max + 2, np.int)
            self.envs = np.copy(self._envs_nan)
            self.output = np.copy(self._output)
            if self.mode == 1:
                self.reset_envs()

            for b in range(self.probY1):
                for a in range(self.probX1):
                    if self._probe1[b][a] == 1:
                        self.envs[self.y1 + b][self.x1 + a] = 1

            self.envs_show = np.copy(self.envs)
            self.num_color1[-1] = self.envs_len1
            self.num_color1[0] = (self.envs == 0).sum()
            self.num_color1[1] = (self.envs == 1).sum()

            if self.time_is_end != 1 and self.steps_is_end != 1:
                self.episode_is_end = 1

                # 5th reward
                self.reward_value1 += 1

                self.steps = 0

        self.num_color_last1 = np.copy(self.num_color1)

    def check2(self):
        # calculate the reward
        self.num_color2[-1] = self.envs_len1
        for n in range(0, self.num_max):
            self.num_color2[n] = (self.envs == n).sum()

        self.num_color2[-2] = self.wafer_len - sum(self.num_color2) + self.num_color2[-2]  # >num_max

        self.dist2 = np.sqrt(np.square(self.y2 - self.y_last2) + np.square(self.x2 - self.x_last2))

        if self.action2 != "None":

            # 1st reward
            if self.num_color_last2[0] - self.num_color2[0] > 0:
                self.reward_value2 += ((self.num_color_last2[0] - self.num_color2[0]) * 0.01)
                if self.num_color_last2[0] - self.num_color2[0] == self.probe_len2:
                    self.reward_value2 += ((self.num_color_last2[0] - self.num_color2[0]) * 0.01)

            # 2nd reward
            for num in range(2, self.num_max + 1):
                if self.num_color2[num] - self.num_color_last2[num] > 0:
                    self.reward_value2 -= (((self.num_color2[num] - self.num_color_last2[num]) * num) * 0.003)

            # 3rd reward
            if np.array_equal(self.num_color2, self.num_color_last2):
                self.reward_value2 -= 0.1

            # 4th reward
            self.reward_value2 -= self.dist2 * 0.01

        # Initialize the environment
        if self.num_color2[0] == 0 or self.time_is_end == 1 or self.steps_is_end == 1:
            self.envs_mean = np.nanmean(self.envs)
            self.envs_std = np.nanstd(self.envs)

            # Stop the screen when the episode is end.
            if self.mode == 1:
                self.build_envs()
                time.sleep(0.1)

            # Initialize the environment
            self.action2 = 'None'
            self.done = 1

            self.y2, self.x2 = self.location2[np.random.randint(len(self.location2))]
            self.y_last2, self.x_last2 = self.y2, self.x2
            self.dist2 = 0

            self.num_color2 = np.zeros(self.num_max + 2, np.int)
            self.envs = np.copy(self._envs_nan)
            self.output = np.copy(self._output)
            if self.mode == 1:
                self.reset_envs()

            for b in range(self.probY2):
                for a in range(self.probX2):
                    if self._probe2[b][a] == 1:
                        self.envs[self.y2 + b][self.x2 + a] = 1

            self.envs_show = np.copy(self.envs)
            self.num_color2[-1] = self.envs_len1
            self.num_color2[0] = (self.envs == 0).sum()
            self.num_color2[1] = (self.envs == 1).sum()

            if self.time_is_end != 1 and self.steps_is_end != 1:
                self.episode_is_end = 1

                # 5th reward
                self.reward_value2 += 1

                self.steps = 0

        self.num_color_last2 = np.copy(self.num_color2)

    def observation(self):
        # produce the image

        probe_list = self.probe_list
        probe_len = self.probe_len
        color = 0

        # first probe
        for c in range(probe_len):
            for num in range(1, self.num_max + 1):
                if self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]] == num:
                    color = NUM_COLORS[num - 1]
            if self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]] > self.num_max:
                color = NUM_COLORS[self.num_max - 1]
            if np.isnan(self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]]):
                color = BUFFER_COLORS
            wafer_check.draw_plt(self.output, self.y_last + self.probe_list[c][0], self.x_last + self.probe_list[c][1],
                                 color)

        for c in range(probe_len):
            color = PROBE_COLORS
            wafer_check.draw_plt(self.output, self.y + self.probe_list[c][0], self.x + self.probe_list[c][1], color)

    def observation1(self):
        # produce the image

        probe_list1 = self.probe_list1
        probe_len1 = self.probe_len1
        color = 0

        # second probe
        for c in range(probe_len1):
            for num in range(1, self.num_max + 1):
                if self.envs[self.y_last1 + probe_list1[c][0]][self.x_last1 + probe_list1[c][1]] == num:
                    color = NUM_COLORS[num - 1]
            if self.envs[self.y_last1 + probe_list1[c][0]][self.x_last1 + probe_list1[c][1]] > self.num_max:
                color = NUM_COLORS[self.num_max - 1]
            if np.isnan(self.envs[self.y_last1 + probe_list1[c][0]][self.x_last1 + probe_list1[c][1]]):
                color = BUFFER_COLORS
            wafer_check.draw_plt(self.output, self.y_last1 + self.probe_list1[c][0],
                                 self.x_last1 + self.probe_list1[c][1], color)

        for c in range(probe_len1):
            color = PROBE_COLORS
            wafer_check.draw_plt(self.output, self.y1 + self.probe_list1[c][0], self.x1 + self.probe_list1[c][1], color)

    def observation2(self):
        # produce the image

        probe_list2 = self.probe_list2
        probe_len2 = self.probe_len2
        color = 0

        # second probe
        for c in range(probe_len2):
            for num in range(1, self.num_max + 1):
                if self.envs[self.y_last2 + probe_list2[c][0]][self.x_last2 + probe_list2[c][1]] == num:
                    color = NUM_COLORS[num - 1]
            if self.envs[self.y_last2 + probe_list2[c][0]][self.x_last2 + probe_list2[c][1]] > self.num_max:
                color = NUM_COLORS[self.num_max - 1]
            if np.isnan(self.envs[self.y_last2 + probe_list2[c][0]][self.x_last2 + probe_list2[c][1]]):
                color = BUFFER_COLORS
            wafer_check.draw_plt(self.output, self.y_last2 + self.probe_list2[c][0],
                                 self.x_last2 + self.probe_list2[c][1], color)

        for c in range(probe_len2):
            color = PROBE_COLORS
            wafer_check.draw_plt(self.output, self.y2 + self.probe_list2[c][0], self.x2 + self.probe_list2[c][1], color)

    def build_envs(self):
        # show the screen

        # first probe
        for c in range(self.probe_len):

            if self.envs[self.y_last + self.probe_list[c][0]][self.x_last + self.probe_list[c][1]] >= 1:
                color = (WHITE / self.num_max).astype(np.int)
            elif np.isnan(self.envs[self.y_last + self.probe_list[c][0]][self.x_last + self.probe_list[c][1]]):
                color = HUR

            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x_last + self.probe_list[c][1]),
                                              (self.y_last + self.probe_list[c][0])))

        for c in range(self.probe_len):
            color = BLUES
            if self.action == 'Invalid':
                color = HUR
            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x + self.probe_list[c][1]),
                                              (self.y + self.probe_list[c][0])))

            # second probe
            for c in range(self.probe_len1):

                if self.envs[self.y_last1 + self.probe_list1[c][0]][self.x_last1 + self.probe_list1[c][1]] >= 1:
                    color = (WHITE / self.num_max).astype(np.int)
                elif np.isnan(self.envs[self.y_last1 + self.probe_list1[c][0]][self.x_last1 + self.probe_list1[c][1]]):
                    color = HUR

                pygame.draw.rect(self.sc,
                                 color,
                                 wafer_check.rect((self.x_last1 + self.probe_list1[c][1]),
                                                  (self.y_last1 + self.probe_list1[c][0])))

            for c in range(self.probe_len1):
                color = BLUES
                if self.action1 == 'Invalid':
                    color = HUR
                pygame.draw.rect(self.sc,
                                 color,
                                 wafer_check.rect((self.x1 + self.probe_list1[c][1]),
                                                  (self.y1 + self.probe_list1[c][0])))

            # third probe
            for c in range(self.probe_len2):

                if self.envs[self.y_last2 + self.probe_list2[c][0]][self.x_last2 + self.probe_list2[c][1]] >= 1:
                    color = (WHITE / self.num_max).astype(np.int)
                elif np.isnan(self.envs[self.y_last2 + self.probe_list2[c][0]][self.x_last2 + self.probe_list2[c][1]]):
                    color = HUR

                pygame.draw.rect(self.sc,
                                 color,
                                 wafer_check.rect((self.x_last2 + self.probe_list2[c][1]),
                                                  (self.y_last2 + self.probe_list2[c][0])))

            for c in range(self.probe_len2):
                color = BLUES
                if self.action2 == 'Invalid':
                    color = HUR
                pygame.draw.rect(self.sc,
                                 color,
                                 wafer_check.rect((self.x2 + self.probe_list2[c][1]),
                                                  (self.y2 + self.probe_list2[c][0])))

        pygame.display.flip()

    def reset_observation(self):
        # reset the image
        color = BUFFER_COLORS
        for row in range(self.envsY):
            for column in range(self.envsX):
                if self._envs[row][column] == -1:
                    wafer_check.draw_plt(self._output, column, row, color)
                    self._envs_nan[row][column] = np.nan

    def reset_envs(self):
        # reset the screen

        self.sc.fill(BLACK)
        for row in range(self.envsY):
            for column in range(self.envsX):
                if self._envs[row][column] == -1:
                    pygame.draw.rect(self.sc, YELLOW, wafer_check.rect(row, column))
                else:
                    pygame.draw.rect(self.sc, BLUE, wafer_check.rect(row, column))

    def action_available(self):
        # evaluate actions that will go beyond the boundary & produce vector to filter

        m = self.envsY
        n = self.envsX
        i = self.probY
        j = self.probX

        for k in range(self.action_space_num):

            act = k % 8
            step = k // 8 + 1

            y = self.y
            x = self.x

            if act == 0:
                if (y + step - 1) < (m - i):
                    y += step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 1:
                if (x + step - 1) < (n - j):
                    x += step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 2:
                if (y - step + 1) > 0:
                    y -= step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 3:
                if (x - step + 1) > 0:
                    x -= step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 4:
                if (y + step - 1) < (m - i) and (x + step - 1) < (n - j):
                    y += step
                    x += step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 5:
                if (y - step + 1) > 0 and (x + step - 1) < (n - j):
                    y -= step
                    x += step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 6:
                if (y - step + 1) > 0 and (x - step + 1) > 0:
                    y -= step
                    x -= step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 7:
                if (y + step - 1) < (m - i) and (x - step + 1) > 0:
                    y += step
                    x -= step
                else:
                    self.available[k] = np.inf
                    continue

            self.available[k] = 0

    def action_available1(self):
        # evaluate actions that will go beyond the boundary & produce vector to filter

        m1 = self.envsY
        n1 = self.envsX
        i1 = self.probY1
        j1 = self.probX1

        for k in range(self.action_space_num):

            act1 = k % 8
            step1 = k // 8 + 1

            y1 = self.y1
            x1 = self.x1

            if act1 == 0:
                if (y1 + step1 - 1) < (m1 - i1):
                    y1 += step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 1:
                if (x1 + step1 - 1) < (n1 - j1):
                    x1 += step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 2:
                if (y1 - step1 + 1) > 0:
                    y1 -= step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 3:
                if (x1 - step1 + 1) > 0:
                    x1 -= step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 4:
                if (y1 + step1 - 1) < (m1 - i1) and (x1 + step1 - 1) < (n1 - j1):
                    y1 += step1
                    x1 += step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 5:
                if (y1 - step1 + 1) > 0 and (x1 + step1 - 1) < (n1 - j1):
                    y1 -= step1
                    x1 += step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 6:
                if (y1 - step1 + 1) > 0 and (x1 - step1 + 1) > 0:
                    y1 -= step1
                    x1 -= step1
                else:
                    self.available[k] = np.inf
                    continue
            elif act1 == 7:
                if (y1 + step1 - 1) < (m1 - i1) and (x1 - step1 + 1) > 0:
                    y1 += step1
                    x1 -= step1
                else:
                    self.available[k] = np.inf
                    continue

            self.available[k] = 0

    def action_available2(self):
        # evaluate actions that will go beyond the boundary & produce vector to filter

        m2 = self.envsY
        n2 = self.envsX
        i2 = self.probY2
        j2 = self.probX2

        for k in range(self.action_space_num):

            act2 = k % 8
            step2 = k // 8 + 1

            y2 = self.y2
            x2 = self.x2

            if act2 == 0:
                if (y2 + step2 - 1) < (m2 - i2):
                    y2 += step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 1:
                if (x2 + step2 - 1) < (n2 - j2):
                    x2 += step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 2:
                if (y2 - step2 + 1) > 0:
                    y2 -= step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 3:
                if (x2 - step2 + 1) > 0:
                    x2 -= step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 4:
                if (y2 + step2 - 1) < (m2 - i2) and (x2 + step2 - 1) < (n2 - j2):
                    y2 += step2
                    x2 += step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 5:
                if (y2 - step2 + 1) > 0 and (x2 + step2 - 1) < (n2 - j2):
                    y2 -= step2
                    x2 += step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 6:
                if (y2 - step2 + 1) > 0 and (x2 - step2 + 1) > 0:
                    y2 -= step2
                    x2 -= step2
                else:
                    self.available[k] = np.inf
                    continue
            elif act2 == 7:
                if (y2 + step2 - 1) < (m2 - i2) and (x2 - step2 + 1) > 0:
                    y2 += step2
                    x2 -= step2
                else:
                    self.available[k] = np.inf
                    continue

            self.available[k] = 0


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    wafer = np.loadtxt('envs.txt')
    probe = np.loadtxt('probe.txt')
    probe1 = np.loadtxt('probe1.txt')

    envs = wafer_check(wafer, probe, probe1, mode=1, training_time=0, training_steps=1000)

    pygame.init()
    pygame.display.set_caption("Wafer Check Simulator")

    # Loop until the user clicks the close button.
    done = False

    while not done:

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP0:
                    envs.reset()
                if event.key == pygame.K_KP2:
                    envs.step(0)
                if event.key == pygame.K_KP6:
                    envs.step(1)
                if event.key == pygame.K_KP8:
                    envs.step(2)
                if event.key == pygame.K_KP4:
                    envs.step(3)
                if event.key == pygame.K_KP3:
                    envs.step(4)
                if event.key == pygame.K_KP9:
                    envs.step(5)
                if event.key == pygame.K_KP7:
                    envs.step(6)
                if event.key == pygame.K_KP1:
                    envs.step(7)
                if event.key == pygame.K_KP5:
                    plt.subplot(1, 2, 1), plt.title('rainbow')
                    plt.imshow(envs.output, cmap='rainbow')
                    plt.subplot(1, 2, 2), plt.title('gray')
                    plt.imshow(envs.output, cmap='gray')
                    plt.show()

    pygame.quit()
