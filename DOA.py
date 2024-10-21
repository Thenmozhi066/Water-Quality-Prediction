import time

import numpy as np


def DOA(drawers,sphere_function,  lb, ub, max_iterations):
    num_drawers, num_variables = drawers.shape[0], drawers.shape[1]
    mutation_rate = 0.1  # Mutation rate
    ct = time.time()
    Convergence = np.zeros(max_iterations)
    fitness = np.zeros(num_drawers)
    # Main loop
    for iteration in range(max_iterations):
        # Evaluate fitness for each drawer
        for i in range(num_drawers):
            fitness[i] = sphere_function(drawers[i, :])

        # Sort drawers based on fitness
        sorted_indices = np.argsort(fitness)
        fitness = fitness[sorted_indices]
        drawers = drawers[sorted_indices, :]

        # Mutation
        for i in range(num_drawers):
            if np.random.rand() < mutation_rate:
                # Select a random variable and mutate it
                variable_index = np.random.randint(0, num_variables)
                mutation = np.random.randn() * (ub - lb) / 10  # Mutation within 10% of the variable's range
                drawers[i, variable_index] += mutation[i,variable_index]

                # Ensure mutated value stays within bounds [lb, ub]
                drawers[i, variable_index] = np.clip(drawers[i, variable_index], lb, ub)[i, variable_index]
        Convergence[iteration] = fitness[0]

    # Display best solution
    best_solution = drawers[0, :]
    best_fitness = fitness[0]
    #     print(f'Iteration {iteration + 1}: Best Fitness = {best_fitness:.6f}')
    #
    # # Display final result
    # print('Optimization finished.')
    # print(f'Best solution found: {best_solution}')
    # print(f'Best fitness: {best_fitness:.6f}')
    ct = time.time() - ct
    return best_fitness, Convergence, best_solution, ct
