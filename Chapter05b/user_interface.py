import tkinter as tk
from tkinter import ttk
import random
from copy import deepcopy

random.seed(42)

window = tk.Tk()
window.title('Sudoku GA Solver')
root = ttk.Frame(window)
root.grid(row=0, column=0)

input_array = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

fields = deepcopy(input_array)

for row in range(9):
    for col in range(9):
        fields[row][col] = tk.StringVar()

prob_label = ttk.Label(root, text='Gib ein Sudoku-Problem ein', font=('arial 40'), justify='center')
prob_label.grid(row=0, column=0, columnspan=3)

quit_button = ttk.Button(root, text='Quit', command=window.destroy, padding=5)
quit_button.grid(row=2, column=2)
fr = ttk.Frame(root)
fr.grid(row=1, column=0, columnspan=3)



superframes = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

for row in range(3):
    for col in range(3):
        superframes[row][col] = ttk.Frame(fr, padding=5)
        superframes[row][col].grid(row=row, column=col)

for row in range(9):
    for column in range(9):
        # fields[row][column] = ttk.Label(fr, text=f'[{row}, {column}]', font=('consolas', 20), anchor='center')
        super_col = column // 3
        super_row = row // 3
        sub_col = column % 3
        sub_row = row % 3

        ttk.Entry(superframes[super_row][super_col], width=3, textvariable=fields[row][column],
                  font=('consolas 30'), justify='center').grid(row=sub_row, column=sub_col)


def show_content():
    for row in range(9):
        for col in range(9):
            value = fields[row][col].get()
            if value != '':
                input_array[row][col] = int(value)


start_button = ttk.Button(root, text='Start', command=show_content, padding=5)
start_button.grid(row=2, column=0)

window.mainloop()

print(input_array)