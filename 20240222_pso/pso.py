#!/usr/bin/env python3

import numpy as np


class Particle:

  def __init__(self):
    return


  def initial_setting(self, config):

    result_dir = config['PSO']['result_dir']
    super().make_directory_rm(result_dir)

    return


  def boundary_setting(self, config):

    # Setting parameter's boundaries

    boundary = config['Bayesian_optimization']['boundary']
    bounds = []
    for n in range(0, len(boundary) ):
      boundary_name = boundary[n]['name']
      parameter_component = boundary[n]['component']
      for m in range(0,  len(parameter_component)):
        bound_type = parameter_component[m]['type']
        bound_min  = parameter_component[m]['bound_min']
        bound_max  = parameter_component[m]['bound_max']
        bounds.append( {'name': bound_type, 'type': 'continuous', 'domain': (bound_min, bound_max) } )
        print( 'Boundary in',bound_type,'component of',boundary_name,'(min--max):', bound_min,'--',bound_max)

    return bounds


  #def objective_function(self, x):
  #  return x[0]**2 + x[1]**2 + x[2]**2 # 例の目的関数 (2次元の場合)
  def objective_function(self, position):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
    return t1 + t2 + t3 + t4


#  def pso(self, config, objective_function, parameter_boundary):
  def pso(self, config):

    dim = 2  # 問題の次元数
    num_particles = 200  # パーティクル数
    max_iter = 100  # 最大イテレーション数

    # Initialize particles
    particle_position = []
    particle_velocity = []
    particle_best_position = []
    particle_best_value = []
    for n in range(0, num_particles):
      particle_position.append( np.random.uniform(low=-30, high=30, size=dim) )
      particle_velocity.append( np.random.uniform(low=-1, high=1, size=dim) )
      particle_best_position.append( particle_position.copy() )
      particle_best_value.append( float('inf') )

    global_best_position = None  # グローバルベスト位置
    global_best_value = float('inf')  # グローバルベストの目的関数値

    # Vizualization
    particle_position_viz = np.zeros(max_iter*num_particles*dim).reshape(max_iter,num_particles,dim)
    particle_velocity_viz = np.zeros(max_iter*num_particles*dim).reshape(max_iter,num_particles,dim)
    particle_solutioin    = np.zeros(max_iter*num_particles*dim).reshape(max_iter,num_particles,dim)

    for i in range(0, max_iter):
      for n in range(0, num_particles):
        # パーティクルの位置の更新
        particle_position[n] += particle_velocity[n]
    
        # パーソナルベストの更新
        value = self.objective_function(particle_position[n])
        if value < particle_best_value[n]:
          particle_best_position[n] = particle_position[n].copy()
          particle_best_value[n] = value
    
        # グローバルベストの更新
        if value < global_best_value:
          global_best_position = particle_position[n].copy()
          global_best_value = value
    
      # パーティクルの速度の更新
      for n in range(0, num_particles):
        inertia = 0.15  # 慣性項
        cognitive_coef = 0.15  # 認知係数
        social_coef = 0.15  # 社会係数
        rand1 = np.random.rand(dim)
        rand2 = np.random.rand(dim)
        cognitive_velocity = cognitive_coef * rand1 * (particle_best_position[n] - particle_position[n])
        social_velocity = social_coef * rand2 * (global_best_position - particle_position[n])
        particle_velocity[n] = inertia * particle_velocity[n] + cognitive_velocity + social_velocity
        particle_position_viz[i,n,:] =  particle_position[n][:]
        particle_velocity_viz[i,n,:] =  particle_velocity[n][:]

    # Tecplot format
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

    filename_tmp='tecplot_particle.dat'
    file_output = open( filename_tmp , 'w')
    header_tmp = "Variables = X, Y, U, V, ID, Solution"  + '\n'
    file_output.write( header_tmp )
#    for n in range(0, num_particles):
#      text_tmp = 'zone i='+str(n+1)+' f=point' + '\n'
#      for i in range(0, max_iter):
#        text_tmp = text_tmp + str(i+1) + ','
#        for m in range(0, 3):
#          text_tmp = text_tmp  + str( particle_position_viz[i,n,m] ) + ', '
#        text_tmp = text_tmp + str(np.linalg.norm(particle_velocity_viz[i,n,:], ord=2)) + '\n'
#      file_output.write( text_tmp )
    for i in range(0, max_iter):
      text_tmp = 'zone t="Time'+str(i+1) +' sec"' + '\n'
      text_tmp =  text_tmp + 'i='+str(num_particles)+' f=point' + '\n'
      for n in range(0, num_particles):
        text_tmp = text_tmp
        for m in range(0,dim):
          text_tmp = text_tmp  + str( particle_position_viz[i,n,m] ) + ', '
        for m in range(0,dim):
          text_tmp = text_tmp  + str( particle_velocity_viz[i,n,m] ) + ', '
        solution_tmp = self.objective_function(particle_position_viz[i,n,:])
        text_tmp = text_tmp + str(n+1) + ',' + str(solution_tmp) + '\n'
      file_output.write( text_tmp )

    file_output.close()

    return global_best_position, global_best_value


  #def drive_optimization(self, config, objective_function, parameter_boundary):
if __name__ == "__main__":
    
  Particle=Particle()
  config = 'dummy'
  #best_position, best_value = Particle.pso(config, objective_function, parameter_boundary)
  best_position, best_value = Particle.pso(config)
  print("Best position:", best_position)
  print("Best value:", best_value)

