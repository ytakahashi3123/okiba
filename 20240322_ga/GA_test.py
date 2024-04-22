import numpy as np


class ga():

  def __init__(self):
    return

  def is_odd(self, i):
    return i % 2 != 0

  # 適応度関数（目的関数）の定義
  #def fitness_function(self, x):
  #  return x**2

  #def fitness_function(self, x):
  #  # Sphere_function
  #  return np.sum(x**2)

  def fitness_function(self, position):
    # Ackley
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
    return t1 + t2 + t3 + t4


  def run_ga(self, config):

    num_generations = 100
    population_size = 100
    mutation_rate = 0.01
    bound_mutation = 0.1
    # Number of dimensions
    num_dim = 2
    # Boundary
    bound_lower_fix = -10
    bound_upper_fix =  10  
    bound_lower = np.ones(num_dim)*bound_lower_fix
    bound_upper = np.ones(num_dim)*bound_upper_fix

    # Odd-even check
    if self.is_odd(population_size):
      print('The population size was odd, so for implementation convenience, it was corrected to an even number')
      population_size = population_size + 1

    # Generating initial population
    population = []
    for i in range(population_size):
      population_tmp = np.random.uniform(low=bound_lower, high=bound_upper, size=num_dim)
      population.append(population_tmp)
    population = np.array(population)

    # History
    population_history = np.zeros(num_generations*population_size*num_dim).reshape(num_generations,population_size,num_dim)
    solution_history   = np.zeros(num_generations*population_size).reshape(num_generations,population_size)

    # Iteration of genetic algorithm
    for n in range(num_generations):
      # Fitness computations (objective function)
      fitness_values = np.zeros(population_size)
      for i in range(population_size):
        fitness_values[i] = self.fitness_function(population[i])
        
      # Selection of the best individual
      best_index = np.argmin(fitness_values)
      best_individual = population[best_index]
      best_fitness = fitness_values[best_index] #np.min(fitness_values)

      # History
      population_history[n,:,:] = population
      solution_history[n,:] = fitness_values

      # Generating a new population (Selection, Crossover, Mutation)
      new_population = []
      for i in range(population_size//2):
        # Tournament selection (Randomly select two individuals and choose the one with the better fitness)
        tournament_indices = np.random.choice(range(population_size), size=2, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        selected_index = tournament_indices[np.argmin(tournament_fitness)]
        selected_individual = population[selected_index]
        
         #Crossover (Take a simple average)
        crossover_individual = np.mean([selected_individual, best_individual], axis=0)
        
        # Mutation (Randomly perturb individuals slightly)
        if np.random.rand() < mutation_rate:
          mutation_amount = np.random.uniform(low=-bound_mutation, high=bound_mutation)
          mutated_individual = selected_individual + mutation_amount
        else:
          mutated_individual = selected_individual
        
        # Adding new individuals to the population
        new_population.extend([crossover_individual, mutated_individual])

      # Update the next generation of individuals
      population = np.array(new_population)

    # Output
    print("Best solution:", best_individual)
    print("Fitness of best solution:", best_fitness)

    # Tecplot
    filename_tmp='tecplota_ga.dat'
    file_output = open( filename_tmp , 'w')
    header_tmp = "Variables = X, Y, ID, Epoch, Solution"  + '\n'
    file_output.write( header_tmp )
    for n in range(0, num_generations):
      text_tmp = 'zone t="Time'+str(n+1) +' sec"' + '\n'
      text_tmp =  text_tmp + 'i='+str(population_size)+' f=point' + '\n'
      for i in range(0, population_size):
        text_tmp = text_tmp
        for j in range(0,num_dim):
          text_tmp = text_tmp  + str( population_history[n,i,j] ) + ', '
        solution_tmp = solution_history[n,i]
        text_tmp = text_tmp + str(i+1) + ', ' + str(n+1) + ', ' + str(solution_tmp) + '\n'
      file_output.write( text_tmp )
    file_output.close()

    return


if __name__ == '__main__':

  ga = ga()
  config = 'dummy'
  ga.run_ga(config)