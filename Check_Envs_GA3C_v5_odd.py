# output=gray colors
import numpy as np
import pygame
import time
from Config import Config

# Define some colors
COLORS = 3
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
BLUE = np.array((60, 150, 255))
PURPLE = np.array((153, 47, 185))
RED_PROBE = np.array((230, 90, 80))
YELLOW = np.array((235, 226, 80))

BACKGROUND_COLORS = 255
BUFFER_COLORS = 170
PROBE_COLORS = 220
OTHER_COLORS = 129

NUM_COLORS = []
for num in range(COLORS):
    NUM_COLORS.append(int(OTHER_COLORS * (1 - num / COLORS)))

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 1
HEIGHT = 1
WIDTH_sc = 10
HEIGHT_sc = 10

# This sets the margin between each cell
MARGIN = 0
MARGIN_sc = 1

# Probe's location when the environment initialize
# odd
gaussian_std = 4
# Initial = [(11, 11)]
Initial = [(2, 2), (11, 11), (2, 11), (11, 2), (8, 5), (5, 8), (8, 8), (5, 5)]

actions = []


class wafer_check():
    def __init__(self, wafer, probe, location=(0, 0), mode=0, training_time=60, training_steps=0):
        self._envs = np.array(wafer)
        self._envs_nan = np.zeros(self._envs.shape)
        self._probe = np.array(probe, np.int)
        self.envsY, self.envsX = self._envs.shape
        self.wafer_len = self.envsY * self.envsX
        self.probY, self.probX = self._probe.shape
        self.probZ = max(self.probY, self.probX)
        self.envs_list = [(b, a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b, a] == -1]
        self.envs_len = len(self.envs_list)
        self.probe_list = [(b, a) for b in range(self.probY) for a in range(self.probX) if self._probe[b, a] == 1]
        self.probe_len = len(self.probe_list)
        self.size = [(self.envsX * WIDTH + (self.envsX + 1) * MARGIN),
                     (self.envsY * HEIGHT + (self.envsY + 1) * MARGIN)]
        self.size1 = [(self.envsX * WIDTH_sc + (self.envsX + 1) * MARGIN_sc),
                      (self.envsY * HEIGHT_sc + (self.envsY + 1) * MARGIN_sc)]
        self._output = np.full((self.size[1], self.size[0]), BACKGROUND_COLORS, np.int)
        self.location = np.array(Initial)
        self.action_space = ['None', 'Down', 'Right', 'Up', 'Left', 'Down-Right', 'Up-Right', 'Up-Left', 'Down-Left']
        self.action_space_num = int((len(self.action_space) - 1) * 1)  # self.probZ)
        self.available = np.zeros(self.action_space_num, dtype=np.float32)
        self.num_max = COLORS
        self.reward_value = 0
        self.envs_mean = None
        self.envs_std = None
        self.mode = mode

        for row in range(self.envsY):
            for column in range(self.envsX):
                if self._envs[row][column] == -1:
                    self._envs_nan[row][column] = np.nan
                elif self._envs[row][column] == 1:
                    self._envs_nan[row][column] = 1

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
        self.y_last, self.x_last = self.y, self.x
        self.steps = 0
        self.dist = 0
        self.total_dist = 0
        self.show_dist = 0
        self.num_color = np.zeros(self.num_max + 2, np.int)
        self.action = 'None'
        self.reward_value = 0
        self.envs = np.copy(self._envs_nan)
        if Config.ODD_TEST:
            self.s = [(b, a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b, a] == 0]
            self.cell = (self._envs == 0).sum()
            self.percent = Config.ODD_PERCENT
            self.ODD = int(self.cell * (self.percent / 100))
            self.odd = np.zeros((self.ODD, 2), dtype=int)
            flag = True
            center_x = np.random.randint(2, self.envsX - 3)
            center_y = np.random.randint(2, self.envsY - 3)
            odd_list = []

            while flag:
                odd = np.zeros((self.ODD, 2), dtype=int)
                if Config.ODD_RANDOM:
                    I = np.random.choice(len(self.s), self.ODD, replace=False)
                    for idx, val in enumerate(I):
                        odd[idx] = self.s[val][0], self.s[val][1]

                else:
                    i = 0
                    while len(odd_list) < self.ODD:
                        y = np.clip(np.int(np.random.randn(1) * gaussian_std + center_y), 2, self.envsY - 3)
                        x = np.clip(np.int(np.random.randn(1) * gaussian_std + center_x), 2, self.envsX - 3)

                        if [y, x] not in odd_list:
                            if self.envs[y][x] == 0:
                                odd_list.append([y, x])
                                odd[i] = y, x
                                i += 1

                # print(odd)

                for row in range(self.envsY):
                    for column in range(self.envsX):
                        if not np.isnan(self.envs[row][column]):
                            self.envs[row][column] = 1

                for i in range(self.ODD):
                    self.envs[odd[i][0]][odd[i][1]] = 0

                if ((self.envs == 0).sum()) == self.ODD:
                    flag = False
                else:
                    self.envs = np.copy(self._envs_nan)

        self.reset_observation()
        self.output = np.copy(self._output)

        if self.mode == 1:
            self.reset_envs()

        for b in range(self.probY):
            for a in range(self.probX):
                if self._probe[b][a] == 1 and not np.isnan(self.envs[self.y + b][self.x + a]):  # fix 0505
                    self.envs[self.y + b][self.x + a] += 1
        self.num_color_last = np.zeros(self.num_max + 2, np.int)
        self.num_color_last[-1] = self.envs_len
        self.num_color_last[0] = (self._envs == 0).sum()
        self.time_end = time.time() + self.training_time
        self.step()
        return self.output, self.available

    def step(self, action=None):
        # Agent's action
        now = time.time()

        if action != None:
            act = ((action) % 8)
            step = int((action) / 8) + 1

        self.done = 0
        self.envs_mean = None
        self.envs_std = None
        self.time_is_end = 0
        self.steps_is_end = 0
        self.episode_is_end = 0
        self.reward_value = 0

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

        elif self.steps >= self.training_steps:
            invalid = -1
            self.steps_is_end = 1

        self.check()

        self.observation()

        self.action_available()

        if self.mode == 1:
            self.build_envs()
            time.sleep(0.01)

        self.y_last = self.y
        self.x_last = self.x

        if self.steps_is_end == 1:
            self.steps = 0

        if self.time_is_end == 1:
            self.steps = 0
            self.time_end = time.time() + self.training_time

        return self.output, self.reward_value, self.done, self.available, self.envs_mean, self.show_dist  # self.envs_std

    def check(self):
        # calculate the reward
        self.num_color[-1] = self.envs_len
        for n in range(0, self.num_max):
            self.num_color[n] = (self.envs == n).sum()

        self.num_color[-2] = self.wafer_len - sum(self.num_color) + self.num_color[-2]  # >num_max

        self.dist = np.sqrt(np.square(self.y - self.y_last) + np.square(self.x - self.x_last))
        self.total_dist += self.dist

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
                time.sleep(0.05)

            # Initialize the environment
            self.action = 'None'
            self.done = 1
            self.y, self.x = self.location[np.random.randint(len(self.location))]
            self.y_last, self.x_last = self.y, self.x
            self.dist = 0
            self.show_dist = self.total_dist
            self.total_dist = 0
            self.num_color = np.zeros(self.num_max + 2, np.int)
            self.envs = np.copy(self._envs_nan)  ##hhh

            if Config.ODD_TEST:
                self.s = [(b, a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b, a] == 0]
                self.cell = (self._envs == 0).sum()
                self.percent = Config.ODD_PERCENT
                self.ODD = int(self.cell * (self.percent / 100))
                self.odd = np.zeros((self.ODD, 2), dtype=int)
                flag = True
                center_x = np.random.randint(2, self.envsX - 3)
                center_y = np.random.randint(2, self.envsY - 3)
                odd_list = []

                while flag:
                    odd = np.zeros((self.ODD, 2), dtype=int)
                    if Config.ODD_RANDOM:
                        I = np.random.choice(len(self.s), self.ODD, replace=False)
                        for idx, val in enumerate(I):
                            odd[idx] = self.s[val][0], self.s[val][1]

                    else:
                        i = 0
                        while len(odd_list) < self.ODD:
                            y = np.clip(np.int(np.random.randn(1) * gaussian_std + center_y), 2, self.envsY - 3)
                            x = np.clip(np.int(np.random.randn(1) * gaussian_std + center_x), 2, self.envsX - 3)

                            if [y, x] not in odd_list:
                                if self.envs[y][x] == 0:
                                    odd_list.append([y, x])
                                    odd[i] = y, x
                                    i += 1

                    for row in range(self.envsY):
                        for column in range(self.envsX):
                            if not np.isnan(self.envs[row][column]):
                                self.envs[row][column] = 1

                    for i in range(self.ODD):
                        self.envs[odd[i][0]][odd[i][1]] = 0

                    if ((self.envs == 0).sum()) == self.ODD:
                        flag = False
                    else:
                        self.envs = np.copy(self._envs_nan)

            self.reset_observation()

            self.output = np.copy(self._output)
            if self.mode == 1:
                self.reset_envs()

            for b in range(self.probY):
                for a in range(self.probX):
                    if self._probe[b][a] == 1:
                        self.envs[self.y + b][self.x + a] += 1

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

    def observation(self):
        # produce the image

        probe_list = self.probe_list
        probe_len = self.probe_len

        for c in range(probe_len):
            for num in range(1, self.num_max + 1):
                if self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]] == num:
                    color = NUM_COLORS[num - 1]
            if self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]] > self.num_max:
                color = NUM_COLORS[-1]
            if np.isnan(self.envs[self.y_last + probe_list[c][0]][self.x_last + probe_list[c][1]]):
                color = BUFFER_COLORS
            wafer_check.draw_plt(self.output, self.y_last + self.probe_list[c][0], self.x_last + self.probe_list[c][1],
                                 color)

        for c in range(probe_len):
            color = PROBE_COLORS
            wafer_check.draw_plt(self.output, self.y + self.probe_list[c][0], self.x + self.probe_list[c][1], color)

    def build_envs(self):
        # show the screen

        for c in range(self.probe_len):

            if self.envs[self.y_last + self.probe_list[c][0]][self.x_last + self.probe_list[c][1]] >= 1:
                color = (WHITE / self.num_max).astype(np.int)
            elif np.isnan(self.envs[self.y_last + self.probe_list[c][0]][self.x_last + self.probe_list[c][1]]):
                color = YELLOW

            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x_last + self.probe_list[c][1]),
                                              (self.y_last + self.probe_list[c][0])))

        for c in range(self.probe_len):
            color = RED_PROBE
            if self.action == 'Invalid':
                color = PURPLE
            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x + self.probe_list[c][1]),
                                              (self.y + self.probe_list[c][0])))

        pygame.display.flip()

    def reset_observation(self):
        # reset the image
        for row in range(self.envsY):
            for column in range(self.envsX):
                if np.isnan(self.envs[row][column]):
                    # self._envs_nan[row][column] = np.nan
                    color = BUFFER_COLORS
                    wafer_check.draw_plt(self._output, row, column, color)
                elif self.envs[row][column] == 0:
                    color = BACKGROUND_COLORS
                    wafer_check.draw_plt(self._output, row, column, color)
                elif self.envs[row][column] == 1:
                    # self._envs_nan[row][column] = 1
                    color = NUM_COLORS[0]
                    wafer_check.draw_plt(self._output, row, column, color)

    def reset_envs(self):
        # reset the screen

        self.sc.fill(BLACK)
        for row in range(self.envsY):
            for column in range(self.envsX):
                if np.isnan(self.envs[row][column]):
                    pygame.draw.rect(self.sc, YELLOW, wafer_check.rect(column, row))
                elif self.envs[row][column] == 1:
                    pygame.draw.rect(self.sc, (WHITE / self.num_max).astype(np.int), wafer_check.rect(column, row))
                else:
                    pygame.draw.rect(self.sc, BLUE, wafer_check.rect(column, row))

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


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    wafer = np.loadtxt('envs.txt')
    probe = np.loadtxt('probe.txt')

    envs = wafer_check(wafer, probe, mode=1, training_time=0, training_steps=1000)

    pygame.init()
    pygame.display.set_caption("Wafer Check Simulator")

    # Loop until the user clicks the close button.
    done = False

    while not done:

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    envs.step(0)
                if event.key == pygame.K_d:
                    envs.step(1)
                if event.key == pygame.K_w:
                    envs.step(2)
                if event.key == pygame.K_a:
                    envs.step(3)
                if event.key == pygame.K_c:
                    envs.step(4)
                if event.key == pygame.K_e:
                    envs.step(5)
                if event.key == pygame.K_q:
                    envs.step(6)
                if event.key == pygame.K_z:
                    envs.step(7)
                if event.key == pygame.K_p:
                    plt.subplot(1, 2, 1), plt.title('rainbow')
                    plt.imshow(envs.output, cmap='rainbow')
                    plt.subplot(1, 2, 2), plt.title('gray')
                    plt.imshow(envs.output, cmap='gray')
                    plt.show()
                    print(actions)
                    print(len(actions))
                # print(envs.num_color)
                # print(envs.reward_value)

    pygame.quit()
