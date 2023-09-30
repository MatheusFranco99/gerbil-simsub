"""Optimization methods: BFGS (Quasi-Newton), Stochastic Gradient Descent"""

import random
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from colors import BLUE, END, PURPLE
from plot import plot_3d_curve, plot_chart, scatter_annotate_x

from simulator import (AVG_LATENCY_METRIC, CONNECTIONS_METRIC, DELIVERY_METRIC, GRAFT_METRIC,
    IHAVE_METRIC, IWANT_METRIC, LATENCY_STD_DEV_METRIC, NUM_MESSAGES_METRIC, Simulator,
    TOTAL_MESSAGES_METRIC)

def bfgs_quasi_newton_optimizer(func: callable,
                                initial_guess: np.array,
                                max_iterations: int = 20,
                                tolerance: float = 1e-6,
                                delta: float | List[float] = 1e-6) -> np.array:
    """Implements BFGS (Quasi-Newton) method"""

    x = initial_guess
    n = len(x)  # Number of variables
    H = np.eye(n)  # Initialize Hessian approximation as the identity matrix.

    if not isinstance(delta,np.ndarray):
        delta = np.ndarray([delta]*len(x))

    delta_norm = np.linalg.norm(delta)

    for iteration in range(max_iterations):
        # Compute the gradient using finite differences.
        grad = np.zeros_like(x)
        for i in range(n):
            x_plus_delta = x.copy()
            x_plus_delta[i] += delta[i]
            grad[i] = (func(x_plus_delta) - func(x)) / delta_norm

        step = -np.dot(H, grad)

        # Line search to find the optimal step size.
        alpha = 1.0

        x_next = x + alpha * step

        # Compute the gradient at the next point.
        grad_next = np.zeros_like(x)
        for i in range(n):
            x_plus_delta = x_next.copy()
            x_plus_delta[i] += delta[i]
            grad_next[i] = (func(x_plus_delta) - func(x_next)) / delta_norm

        y = grad_next - grad
        s = x_next - x

        y = np.transpose(y)
        s = np.transpose(s)

        # BFGS update formula for Hessian approximation.
        H = H + np.outer(y, y) / np.dot(y, s) - np.dot(np.dot(H, s), np.dot(s.transpose(), H)) / np.dot(np.dot(s, H), s)

        x = x_next

        # Convergence check based on the norm of the gradient.
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            return x

    print("Maximum iterations reached.")

    return x



def stochastic_gradient_descent(func: callable,
                                initial_guess: np.array,
                                indices_to_update: list,
                                steps: list,
                                min_bounds: list,
                                max_bounds: list,
                                num_iterations: int = 20
                                ) -> (np.array,float,List[np.array],List[float]):
    """Implements Stochastic Gradient Descent method"""

    # Stores tested values and their costs during optimization
    tested_x_vectors = []
    testes_costs = []

    # Iteration counter
    iteration = 1

    # Random perturbation for stochastic purposes
    def coin() -> int:
        return [-1,0,1][random.randint(0,2)]

    # Current solution
    current_x = initial_guess.copy()

    # Current cost
    current_cost = func(current_x)

    tested_x_vectors += [current_x]
    testes_costs += [current_cost]


    while iteration <= num_iterations:

        # Perturb variables for given indices and steps
        x_perturbed = current_x.copy()
        for idx in indices_to_update:
            x_perturbed[idx] += coin() * steps[idx]

        # Enforce bounds on variables
        for idx, v in enumerate(x_perturbed):
            x_perturbed[idx] = min(max_bounds[idx], v)
            x_perturbed[idx] = max(min_bounds[idx], v)

        # Simulate
        new_cost = func(x_perturbed)

        tested_x_vectors += [x_perturbed]
        testes_costs += [new_cost]

        # Update solution if better
        if new_cost < current_cost:
            current_x = x_perturbed

            current_cost = new_cost

        iteration += 1

    return current_x, current_cost, tested_x_vectors,testes_costs




def exhaustive_analysis(simulator: Simulator,
                        initial_vector: np.array,
                        cost_function: callable,
                        variation: Dict[int,Dict[str,List]],
                        repeat: int | None = None,
                        verbose = True) -> None:
    """ Performs exhaustive analysis
        - The 'variations' argument should be the following dictionary:
            - variable index in vector (that will to be varied) -> {'name': 'VARIABLE NAME', 'values': [VARIABLE VALUES]}
    """

    if repeat is None:
        repeat = 1

    def log(txt: str, separator = False, tab = True):
        """Abstract loggeing"""
        if separator:
            delimiter = " " + "="*20 + " "
            txt = delimiter + txt + delimiter
        if tab:
            txt = "\t" + txt
        print(txt)

    # Stores simulation result: x vector -> {'Metrics': metrics, 'Cost': cost}
    data = {}

    # Auxiliary function
    def run_and_record_simulation(simulator: Simulator, x_vector: np.ndarray, repeat: int | None = None, verbose=True) -> dict:
        """Run the simulator and returns metrics and cost"""
        if verbose:
            log(f"Simulating {BLUE}f({x_vector}){END}",separator = True,tab = False)

        result = simulator.run_simulation(x_vector, repeat = repeat)

        metrics_avg, metrics_stdev = simulator.extract_metrics(result)
        if verbose:
            log(f"Metrics: {PURPLE}{metrics_avg}{END}")

        cost = cost_function(metrics_avg,verbose=verbose, log=log)

        return {"Metrics":metrics_avg, "Metrics Stdev": metrics_stdev,"Cost":cost}

    # For each variable index, runs the simulation with its list of values
    for x1_idx in variation.keys():
        for x1_v in variation[x1_idx]['values']:
            # Adjust vector
            x_vector = initial_vector.copy()
            x_vector[x1_idx] = x1_v

            data[str(x_vector)] = run_and_record_simulation(simulator,x_vector,verbose = verbose, repeat = repeat)

    # For each combination of two variables, runs the simulation
    for x1_idx in variation.keys():
        for x2_idx in variation.keys():

            if x1_idx == x2_idx or x1_idx > x2_idx:
                continue

            for x1_v in variation[x1_idx]['values']:
                for x2_v in variation[x2_idx]['values']:
                    x_vector = initial_vector.copy()
                    x_vector[x1_idx] = x1_v
                    x_vector[x2_idx] = x2_v

                    data[str(x_vector)] = run_and_record_simulation(simulator,x_vector,verbose = verbose, repeat = repeat)


    # Results

    # For each variable, plot variation impact
    for x_idx in variation.keys():

        x = variation[x_idx]['values']

        x_vector_values = []
        for x_v in variation[x_idx]['values']:
            x_vector = initial_vector.copy()
            x_vector[x_idx] = x_v
            x_vector_values += [x_vector]

        graft = [data[str(x_vector)]['Metrics'][GRAFT_METRIC] for x_vector in x_vector_values]
        graft_stdev = [data[str(x_vector)]['Metrics Stdev'][GRAFT_METRIC] for x_vector in x_vector_values]
        ihave = [data[str(x_vector)]['Metrics'][IHAVE_METRIC] for x_vector in x_vector_values]
        ihave_stdev = [data[str(x_vector)]['Metrics Stdev'][IHAVE_METRIC] for x_vector in x_vector_values]
        iwant = [data[str(x_vector)]['Metrics'][IWANT_METRIC] for x_vector in x_vector_values]
        iwant_stdev = [data[str(x_vector)]['Metrics Stdev'][IWANT_METRIC] for x_vector in x_vector_values]
        number_of_messages = [data[str(x_vector)]['Metrics'][NUM_MESSAGES_METRIC] for x_vector in x_vector_values]
        number_of_messages_stdev = [data[str(x_vector)]['Metrics Stdev'][NUM_MESSAGES_METRIC] for x_vector in x_vector_values]
        connections = [data[str(x_vector)]['Metrics'][CONNECTIONS_METRIC] for x_vector in x_vector_values]
        connections_stdev = [data[str(x_vector)]['Metrics Stdev'][CONNECTIONS_METRIC] for x_vector in x_vector_values]
        delivery = [data[str(x_vector)]['Metrics'][DELIVERY_METRIC] for x_vector in x_vector_values]
        delivery_stdev = [data[str(x_vector)]['Metrics Stdev'][DELIVERY_METRIC] for x_vector in x_vector_values]
        latencies = [data[str(x_vector)]['Metrics'][AVG_LATENCY_METRIC] for x_vector in x_vector_values]
        latencies_stdev = [data[str(x_vector)]['Metrics Stdev'][AVG_LATENCY_METRIC] for x_vector in x_vector_values]
        latency_stddev = [data[str(x_vector)]['Metrics'][LATENCY_STD_DEV_METRIC] for x_vector in x_vector_values]
        latency_stddev_stdev = [data[str(x_vector)]['Metrics Stdev'][LATENCY_STD_DEV_METRIC] for x_vector in x_vector_values]
        costs = [data[str(x_vector)]['Cost'] for x_vector in x_vector_values]
        total_messages = [data[str(x_vector)]['Metrics'][TOTAL_MESSAGES_METRIC] for x_vector in x_vector_values]
        total_messages_stdev = [data[str(x_vector)]['Metrics Stdev'][TOTAL_MESSAGES_METRIC] for x_vector in x_vector_values]

        # Variaiton impact
        x_label = variation[x_idx]['name']

        _, ax = plt.subplots(5, 2, figsize=(18,10))

        plot_chart(x, graft, error = graft_stdev, x_label = x_label, y_label = GRAFT_METRIC, ax = ax[0,0], savefig = False, showfig = False)
        plot_chart(x, ihave, error = ihave_stdev, x_label = x_label, y_label = IHAVE_METRIC, ax = ax[0,1], savefig = False, showfig = False)
        plot_chart(x, iwant, error = iwant_stdev, x_label = x_label, y_label = IWANT_METRIC, ax = ax[1,0], savefig = False, showfig = False)
        plot_chart(x, number_of_messages, error = number_of_messages_stdev, x_label = x_label, y_label = NUM_MESSAGES_METRIC, ax = ax[1,1], savefig = False, showfig = False)
        plot_chart(x, connections, error = connections_stdev, x_label = x_label, y_label = CONNECTIONS_METRIC, ax = ax[2,0], savefig = False, showfig = False)
        plot_chart(x, delivery, error = delivery_stdev, x_label = x_label, y_label = DELIVERY_METRIC, ax = ax[2,1], savefig = False, showfig = False)
        plot_chart(x, latencies, error = latencies_stdev, x_label = x_label, y_label = AVG_LATENCY_METRIC, ax = ax[3,0], savefig = False, showfig = False)
        plot_chart(x, latency_stddev, error = latency_stddev_stdev, x_label = x_label, y_label = LATENCY_STD_DEV_METRIC, ax = ax[3,1], savefig = False, showfig = False)
        plot_chart(x, costs, x_label = x_label, y_label = 'Costs', ax = ax[4,0], savefig = False, showfig = False)
        plot_chart(x, total_messages, error = total_messages_stdev, x_label = x_label, y_label = TOTAL_MESSAGES_METRIC, ax = ax[4,1], filename = f"Variation impact of {x_label}.png", savefig = True, showfig = True)

        # Combined impact
        scatter_annotate_x(x,latencies,total_messages,AVG_LATENCY_METRIC,TOTAL_MESSAGES_METRIC, y_error=latencies_stdev, z_error=total_messages_stdev, sections = True, savefig = True, showfig = True)
        scatter_annotate_x(x,latencies,number_of_messages,AVG_LATENCY_METRIC,NUM_MESSAGES_METRIC, y_error=latencies_stdev, z_error=number_of_messages_stdev,sections = True, savefig = True, showfig = True)
        scatter_annotate_x(x,latencies,iwant, AVG_LATENCY_METRIC,IWANT_METRIC,y_error=latencies_stdev, z_error=iwant_stdev, sections = True, savefig = True, showfig = True)

        plot_3d_curve(np.array(latencies), np.array(number_of_messages), iwant, AVG_LATENCY_METRIC, NUM_MESSAGES_METRIC, IWANT_METRIC, x_error= latencies_stdev, y_error= number_of_messages_stdev, z_error= iwant_stdev,  annotation=x, savefig=True, showfig=True)



    # For each combination of two variables, plot combined variation impact
    for x1_idx in variation.keys():
        for x2_idx in variation.keys():

            if x1_idx == x2_idx or x1_idx > x2_idx:
                continue

            # Gets combination tuples and list of x_vectors for all combinations
            x_tuples = []
            x_vector_values = []
            for x1_v in variation[x1_idx]['values']:
                for x2_v in variation[x2_idx]['values']:
                    x_vector = initial_vector.copy()
                    x_vector[x1_idx] = x1_v
                    x_vector[x2_idx] = x2_v

                    x_tuples += [(x1_v, x2_v)]
                    x_vector_values += [x_vector]

            # Get metrics results for list of x_vectors
            graft = [data[str(x_vector)]['Metrics'][GRAFT_METRIC] for x_vector in x_vector_values]
            ihave = [data[str(x_vector)]['Metrics'][IHAVE_METRIC] for x_vector in x_vector_values]
            iwant = [data[str(x_vector)]['Metrics'][IWANT_METRIC] for x_vector in x_vector_values]
            number_of_messages = [data[str(x_vector)]['Metrics'][NUM_MESSAGES_METRIC] for x_vector in x_vector_values]
            connections = [data[str(x_vector)]['Metrics'][CONNECTIONS_METRIC] for x_vector in x_vector_values]
            delivery = [data[str(x_vector)]['Metrics'][DELIVERY_METRIC] for x_vector in x_vector_values]
            latencies = [data[str(x_vector)]['Metrics'][AVG_LATENCY_METRIC] for x_vector in x_vector_values]
            latency_stddev = [data[str(x_vector)]['Metrics'][LATENCY_STD_DEV_METRIC] for x_vector in x_vector_values]
            costs = [data[str(x_vector)]['Cost'] for x_vector in x_vector_values]
            total_messages = [data[str(x_vector)]['Metrics'][TOTAL_MESSAGES_METRIC] for x_vector in x_vector_values]

            # Plot scatter plots with annotations
            scatter_annotate_x(x_tuples,latencies,total_messages,AVG_LATENCY_METRIC,TOTAL_MESSAGES_METRIC, sections = True, savefig = True, showfig = True)
            scatter_annotate_x(x_tuples,latencies,number_of_messages,AVG_LATENCY_METRIC,NUM_MESSAGES_METRIC, sections = True, savefig = True, showfig = True)
            scatter_annotate_x(x_tuples,latencies,iwant, AVG_LATENCY_METRIC,IWANT_METRIC, sections = True, savefig = True, showfig = True)

            # Plot 3D curves
            x1_label = variation[x1_idx]['name']
            x2_label = str(variation[x2_idx]['name'])

            def plot_3d(y, y_label):
                return plot_3d_curve(np.array([x[0] for x in x_tuples]),np.array([x[1] for x in x_tuples]),y,
                                    x1_label, x2_label, y_label,
                                    savefig=True, showfig=True)

            plot_3d(graft, GRAFT_METRIC)
            plot_3d(ihave, IHAVE_METRIC)
            plot_3d(iwant, IWANT_METRIC)
            plot_3d(connections, CONNECTIONS_METRIC)
            plot_3d(delivery, DELIVERY_METRIC)
            plot_3d(number_of_messages, NUM_MESSAGES_METRIC)
            plot_3d(latencies, AVG_LATENCY_METRIC)
            plot_3d(latency_stddev, LATENCY_STD_DEV_METRIC)

            plot_3d_curve(np.array(latencies), np.array(number_of_messages), iwant,
                          AVG_LATENCY_METRIC, NUM_MESSAGES_METRIC, IWANT_METRIC,
                          annotation=x_tuples, savefig=True, showfig=True)
            plot_3d_curve(np.array(latencies), np.array(total_messages), iwant,
                          AVG_LATENCY_METRIC, TOTAL_MESSAGES_METRIC, IWANT_METRIC,
                          annotation=x_tuples, savefig=True, showfig=True)


    # Get all tested x_vectors and costs
    all_x_tuples = data.keys()
    all_costs = [data[k]['Cost'] for k in all_x_tuples]

    # Sort
    all_x_tuples = [y for _,y in sorted(zip(all_costs,all_x_tuples))]
    all_costs = sorted(all_costs)

    # Present 10 best results
    log("Best costs",separator=True,tab=False)
    for i in range(min(10,len(all_x_tuples))):
        log(f"{all_costs[i]} - {all_x_tuples[i]}")
