import autoit
import time
import keyboard
import random


def boot_pico8_celeste():
    PICO8_PATH = "\"C:\\Program Files (x86)\\PICO-8\\pico8.exe\""
    
    if autoit.win_exists("[REGEXPTITLE:.*PICO-8.*]"):
        autoit.win_close("[REGEXPTITLE:.*PICO-8.*]")
    if not autoit.win_exists("[REGEXPTITLE:.*PICO-8.*]"):
        print("Opening PICO-8...")
        autoit.run(PICO8_PATH)
        try:
            autoit.win_wait_active("[REGEXPTITLE:.*PICO-8.*]", 90)
        except autoit.AutoItError:
            print("Cannot open PICO-8.")
    if autoit.win_exists("[REGEXPTITLE:.*PICO-8.*]"):
        autoit.win_activate("[REGEXPTITLE:.*PICO-8.*]")
        print("PICO-8 running")
    
    
    hwnd = autoit.win_get_handle("[REGEXPTITLE:.*PICO-8.*]")
    print(hwnd)
    autoit.win_move_by_handle(hwnd, -8, 0, 144, 167)
    autoit.mouse_click("left", 50, 50)
    
    
    time.sleep(1.5)
    keyboard.write("cd ..\n")
    time.sleep(0.2)
    keyboard.write("install_games\n")
    time.sleep(0.2)
    keyboard.write("cd games\n")
    time.sleep(0.2)
    keyboard.write("load celeste\n")
    time.sleep(0.2)
    keyboard.write("run\n")
    time.sleep(0.5)
    keyboard.press('c')
    time.sleep(0.1)
    keyboard.release('c')
    time.sleep(3)
    

def play_random():
    h_arrows = ["left arrow", "right arrow", ""]
    v_arrows = ["up arrow", "down arrow", ""]
    buttons = ['x', 'z', '']
    for i in range(1000):
        txt = ''
        h = random.choice(h_arrows)
        if h:
            txt = h
        v = random.choice(v_arrows)
        if v:
            if txt:
                txt += '+'
            txt += v
        b = random.choice(buttons)
        if b:
            if txt:
                txt += '+'
            txt += b
        if txt:
            keyboard.send(txt)
            time.sleep(1/30)

boot_pico8_celeste()
play_random()