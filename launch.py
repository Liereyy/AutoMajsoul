import time

import pyautogui

from action.action import *
from strategy import *

gui = GUIInterface()

# time.sleep(2)
# gui.calibrateMenu()

if True:
    cnt = 0
    while 1:
        if cnt == 20:
            cnt = 0
        cnt += 1

        gui.flush(cnt == 19)  # 周期性点击
        gui.run()
else:
    gui.flush(False)  # 周期性点击一下
    gui.run()
