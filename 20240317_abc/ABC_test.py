
import numpy as np

class abc():

  def __init__(self):
    return


  def initial_setting(self, config):

    return


  def boundary_setting(self, config):

    # Setting parameter's boundaries

    boundary = config['parameter_optimized']['boundary']
    parameter_boundary = []
    for n in range(0, len(boundary) ):
      parameter_component = boundary[n]['component']
      for m in range(0, len(parameter_component)):
        parameter_boundary.append( (parameter_component[m]['bound_min'],parameter_component[m]['bound_max']) )

    return parameter_boundary


  # 問題関数を定義
#  def objective_function(self, x):
#    return x[0]**2 + x[1]**2 

  def objective_function(self, position):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
    return t1 + t2 + t3 + t4


  def generate_food_source(self, bound_low, bound_high, num_dim):
    #food_source = bound_low + np.random.uniform(low=bound_low, high=bound_high, size=num_dim)
    food_source = np.random.uniform(low=bound_low, high=bound_high, size=num_dim)
    #food_source = np.array( [np.random.uniform(low=bound_low, high=bound_high, size=num_dim )] )
    return food_source


  def fitness_value_function(self, solution):
    # Solution: Solution of objective function
    if solution >= 0:
      fitness = 1.0/(1.0+solution)
    else: 
      fitness = 1.0+abs(solution)
    return fitness


  def roulette_wheel_selection(self, fitness_values):
    # Calculate the total sum of the fitness values of each individual
    total_fitness = sum(fitness_values)  
    # Select a random position on the roulette wheel
    selected_point = np.random.uniform(0, total_fitness)
    # Locate the individual corresponding to the selected position
    cumulative_fitness = 0
    for i, fitness in enumerate(fitness_values):
      cumulative_fitness += fitness
      if cumulative_fitness >= selected_point:
        return i


  def run_abc(self, config):

    # Number of iteration
    num_optiter = 100 
    # Number of employed bees 
    num_employ_bees = 80
    # Number of onlooking bees 
    num_onlook_bees = 20
    # Limit of visit
    vist_limit = 30
    # Number of dimensions
    num_dim = 2

    # Visit counter
    visit_counter = np.zeros(num_employ_bees).astype(int)
    # Boundary
    bound_lower_fix = -10
    bound_upper_fix =  10  
    bound_lower = np.ones(num_dim)*bound_lower_fix
    bound_upper = np.ones(num_dim)*bound_upper_fix

    # Best solutioin
    #best_food_source = float('inf')
    best_solution = float('inf')

    best_food_source_hisotry = np.zeros(num_optiter*num_dim).reshape(num_optiter,num_dim)
    best_solutioin_hisotry   = np.zeros(num_optiter)
    best_index_hisotry       = np.zeros(num_optiter, dtype=int)
    food_source_history      = np.zeros(num_optiter*num_employ_bees*num_dim).reshape(num_optiter,num_employ_bees,num_dim)
    solution_history         = np.zeros(num_optiter*num_employ_bees).reshape(num_optiter,num_employ_bees)

    # Initialization phase
    food_source = []
    solution = []
    for i in range(num_employ_bees):
      food_source_tmp = self.generate_food_source(bound_lower, bound_upper, num_dim)
      food_source.append( food_source_tmp )
      solution.append( self.objective_function( food_source_tmp ) )

    # Iteration
    for n in range(num_optiter):

      # Employed bee phase
      for i in range(num_employ_bees):
        phi = 2.0*np.random.rand(num_dim) - 1.0
        index = np.random.randint(num_employ_bees-1)
        food_source_new = food_source[i] + phi*( food_source[i] - food_source[index] )
        solution_new = self.objective_function(food_source_new)
        # Update source
        if self.fitness_value_function( solution_new ) > self.fitness_value_function( solution[i] ):
          food_source[i] = food_source_new
          solution[i]    = solution_new
          visit_counter[i] = 0
        else:
          visit_counter[i] += 1

      # Onlooker bee phase
      fitness_values = []
      for i in range(num_employ_bees):
        fitness_values.append( self.fitness_value_function( solution[i] ) )
      for i in range(num_onlook_bees):
        # Select randomly according to the evaluation value of the food source
        index = self.roulette_wheel_selection( fitness_values )
        # The acquisition count of the food source +1
        visit_counter[index] += 1

      # Scout bee phase
      for i in range(num_employ_bees):
        # Replace the food sources that have been visited more than a certain number of times
        if visit_counter[i] > vist_limit:
          food_source[i] = self.generate_food_source(bound_lower, bound_upper, num_dim)
          solution[i] = self.objective_function(food_source[i])
          visit_counter[i] = 0

      # Update best solution
      for i in range(num_employ_bees):
        if best_solution > solution[i] :
          best_food_source = food_source[i]
          best_solution = solution[i]

      # History
      food_source_history[n,:,:] = food_source
      solution_history[n,:] = solution
      min_index = np.argmin(solution)
      best_index_hisotry[n] = min_index
      best_food_source_hisotry[n,:] = food_source[min_index]
      best_solutioin_hisotry[n] = solution[min_index]

      print(n, best_food_source, best_solution)

    # Output
    print("最小値:", best_food_source )
    print("最小値の関数値:", best_solution )

    # Tecplot
    filename_tmp='tecplota_abc.dat'
    file_output = open( filename_tmp , 'w')
    header_tmp = "Variables = X, Y, ID, Epoch, Solution"  + '\n'
    file_output.write( header_tmp )
    for n in range(0, num_optiter):
      text_tmp = 'zone t="Time'+str(n+1) +' sec"' + '\n'
      text_tmp =  text_tmp + 'i='+str(num_employ_bees)+' f=point' + '\n'
      for i in range(0, num_employ_bees):
        text_tmp = text_tmp
        for j in range(0,num_dim):
          text_tmp = text_tmp  + str( food_source_history[n,i,j] ) + ', '
        solution_tmp = solution_history[n,i]
        text_tmp = text_tmp + str(i+1) + ', ' + str(n+1) + ', ' + str(solution_tmp) + '\n'
      file_output.write( text_tmp )
    file_output.close()

    # Objective function
    filename_tmp='tecplot_function.dat'
    file_output = open( filename_tmp , 'w')
    header_tmp = "Variables = X, Y, Solution"  + '\n'
    file_output.write( header_tmp )
    x_len = 100
    y_len = 100
    x_ref = np.linspace(-30,30,x_len)
    y_ref = np.linspace(-30,30,y_len)
    text_tmp = 'zone t="Function_ref"' + '\n'
    text_tmp =  text_tmp + 'i='+str(x_len)+' j='+str(y_len)+' f=point' + '\n'
    for j in range(0, y_len):
      for i in range(0, x_len):
        solution_tmp = self.objective_function( np.array([x_ref[i], y_ref[j]]) )
        text_tmp = text_tmp + str(x_ref[i]) + ',' + str(y_ref[j]) + ',' + str(solution_tmp) + '\n'
    file_output.write( text_tmp )
    file_output.close()

    return


if __name__ == '__main__':

  abc = abc()
  config = 'dummy'
  abc.run_abc(config)