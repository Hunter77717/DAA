from flask import Flask, render_template, request
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def brute_force_assignment(cost_matrix):
    n = len(cost_matrix)
    min_cost = float('inf')
    best_assignment = None

    for perm in permutations(range(n)):
        cost = sum(cost_matrix[i][p] for i, p in enumerate(perm))
        if cost < min_cost:
            min_cost = cost
            best_assignment = perm

    return best_assignment, min_cost

def hungarian_assignment(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    assignment = list(zip(row_ind, col_ind))
    return assignment, total_cost

@app.route('/solve', methods=['POST'])
def solve():
    matrix = request.form.getlist('cost')
    algorithm = request.form.get('algorithm')

    n = int(len(matrix) ** 0.5)
    cost_matrix = np.array(matrix, dtype=int).reshape((n, n))

    start_time = time.time()

    if algorithm == 'brute':
        best_assignment, total_cost = brute_force_assignment(cost_matrix)
        assignments = [(i+1, best_assignment[i]+1, cost_matrix[i][best_assignment[i]]) for i in range(n)]
        complexities = {
            "time_best": f"O(n!)",
            "time_worst": f"O(n!)",
            "space": f"O(n)"
        }
        method_display = "Brute Force"
    else:  # Hungarian
        assignment, total_cost = hungarian_assignment(cost_matrix)
        assignments = [(i+1, j+1, cost_matrix[i][j]) for i, j in assignment]
        complexities = {
            "time_best": f"O(n³)",
            "time_worst": f"O(n³)",
            "space": f"O(n²)"
        }
        method_display = "Hungarian Method"

    end_time = time.time()
    runtime = round(end_time - start_time, 4)

    return render_template('index.html',
                           result=assignments,
                           total=total_cost,
                           complexities=complexities,
                           runtime=runtime,
                           method_display=method_display)

if __name__ == '__main__':
    app.run(debug=True)
