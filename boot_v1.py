import autoit
from pynput.keyboard import Key, Controller

CELESTE_PATH = "\"C:\\Program Files (x86)\\Steam\\steamapps\\common\\Celeste\\Celeste.exe\""

if not autoit.win_exists("[TITLE:Celeste]"):
    print("Opening Celeste...")
    autoit.run(CELESTE_PATH)
    try:
        autoit.win_wait_active("[TITLE:Celeste]", 90)
    except autoit.AutoItError:
        print("Cannot open Celeste.")
if autoit.win_exists("[TITLE:Celeste]"):
    print("Celeste running")
    autoit.win_activate("[TITLE:Celeste]")


hwnd = autoit.win_get_handle("[TITLE:Celeste]")
print(hwnd)
autoit.win_move_by_handle(hwnd, -8, 0, 336, 219)
autoit.mouse_click("left", 50, 50)


keyboard = Controller()
keyboard.press('c')
keyboard.release('c')