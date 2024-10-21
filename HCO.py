import time
import numpy as np


def evaluate_fitness(position, objective_function):
    return objective_function(position)


# Hermit Crab Optimizer (HCO)

def HCO(population, objective_function, lower_bound, upper_bound, max_generations):
    num_hermit_crabs, num_dimensions = population.shape[0], population.shape[1]
    convergence = np.zeros(max_generations)
    global_best_position = None
    global_best_fitness = float('inf')
    fitness = np.zeros(num_hermit_crabs)
    for n in range(num_hermit_crabs):
        fitness[n] = evaluate_fitness(population[n], objective_function)
    ct = time.time()
    for generation in range(max_generations):
        for i, hermit_crab_position in enumerate(population):
            fitness[i] = evaluate_fitness(hermit_crab_position, objective_function)
            if fitness[i] < global_best_fitness:
                global_best_position = hermit_crab_position.copy()
                global_best_fitness = np.min(fitness)

            # Update velocity
            r = np.random.random()
            velocity = np.random.uniform(r, 1, num_dimensions) * (
                    population[np.random.randint(num_hermit_crabs)] - hermit_crab_position) + \
                       np.random.uniform(-1, 1, num_dimensions) * (global_best_position - hermit_crab_position)
            # Update position
            new_position = hermit_crab_position + velocity
            # Bound position within the search space
            population[i] = np.clip(new_position, lower_bound[i], upper_bound[i])
        convergence[generation] = np.min(fitness)
    ct = time.time() - ct
    return global_best_fitness, convergence, global_best_position, ct
