import numpy as np

X = [1, 2]
state = [0.0, 0.0]

w_cell_state = np.array([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.array([0.5, 0.6])
b_cell = np.array([0.1, -0.1])

w_output = np.array([[1.0], [2.0]])
b_output = 0.1


for i in range(len(X)):
    before_act = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_act)
    final_output = np.dot(state, w_output) + b_output

    print(f'before activation: {before_act}')
    print(f'state: {state}')
    print(f'output: {final_output}')

