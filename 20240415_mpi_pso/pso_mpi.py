#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI


class mpi():

  def __init__(self):
    self.MPI = MPI
    self.comm = MPI.COMM_WORLD
    self.rank = MPI.COMM_WORLD.Get_rank()
    self.size = MPI.COMM_WORLD.Get_size()
    self.name = MPI.Get_processor_name()
    if self.size == 1:
      self.flag_mpi = False
    else :
      self.flag_mpi = True
    print('Rank:',self.rank, ', Num process:',self.size, ', Name:',self.name)
    return

  # Decorator for time measurement
  def mpitime_measurement_decorated(func):
    @wraps(func)
    def wrapper(*args, **kargs) :
      #text_blue = '\033[94m'
      #text_green = '\033[92m'
      text_yellow = '\033[93m'
      text_end = '\033[0m'
      flag_time_measurement = False
      if flag_time_measurement :
        start_time = self.MPI.time()
        result = func(*args,**kargs)
        elapsed_time = self.MPI.time() - start_time
        if self.rank == 0:
          print('Elapsed time of '+str(func.__name__)+str(':'),text_yellow + str(elapsed_time) + text_end,'s')
      else :
        result = func(*args,**kargs)
      return result 
    return wrapper
    


class optimizer_pso:

  def __init__(self):
    return


  def objective_function(self, x):
    return np.sum(x**2)  # 例の目的関数 (2次元の場合)

  #def objective_function(self, position):
  #  t1 = 20
  #  t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
  #  t3 = np.e
  #  t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
  #  return t1 + t2 + t3 + t4


  def pso(self,mpi_instance):

    # 計算パラメータ
    dim = 2  # 問題の次元数
    max_iter = 100  # 最大イテレーション数
    num_particles = 200  # パーティクル数

    # PSOにおける粒子パラメータ
    inertia = 0.5  # 慣性項
    cognitive_coef = 1.5  # 認知係数
    social_coef = 1.5  # 社会係数

    # MPI process: プロセスごとに担当するループ範囲を分割
    flag_mpi = mpi_instance.flag_mpi
    if flag_mpi :
      MPI = mpi_instance.MPI
      comm = mpi_instance.comm
      size = mpi_instance.size
      rank = mpi_instance.rank
      chunk_size = num_particles // size
      start = rank * chunk_size
      end = (rank + 1) * chunk_size if rank < size - 1 else num_particles
      print('Rank, Iter_start, Iter_zend', rank, start, end)
    else :
      rank = 0
      num_particle_start = 0
      num_particle_end = num_particle

    # 粒子座標や速度の初期化
    particle_position = []
    particle_velocity = []
    particle_best_position = []
    particle_best_value = []
    for n in range(0, num_particles):
      particle_position.append( np.random.uniform(low=-30, high=30, size=dim) )
      particle_velocity.append( np.random.uniform(low=-1, high=1, size=dim) )
      particle_best_position.append( particle_position.copy() )
      particle_best_value.append( float('inf') )

    # History
    global_best_index_hisotry = np.zeros(max_iter, dtype=int)

    # 可視化用変数
    particle_position_viz = np.zeros(max_iter*num_particles*dim).reshape(max_iter,num_particles,dim)
    particle_velocity_viz = np.zeros(max_iter*num_particles*dim).reshape(max_iter,num_particles,dim)
    particle_solution     = np.zeros(max_iter*num_particles).reshape(max_iter,num_particles)
    particle_rank         = np.zeros(num_particles).reshape(num_particles)

    for i in range(0, max_iter):
      global_best_position = None  # グローバルベスト位置
      global_best_value = float('inf')  # グローバルベストの目的関数値
      for n in range(start, end):    
        # パーソナルベストの更新
        value = self.objective_function(particle_position[n])
        particle_solution[i,n] = value
        if value < particle_best_value[n]:
          particle_best_position[n] = particle_position[n].copy()
          particle_best_value[n] = value
    
        # グローバルベストの更新
        if value < global_best_value:
          global_best_position = particle_position[n].copy()
          global_best_value = value
          global_best_index_hisotry[i] = n

      # MPI process
      if flag_mpi :
        # プロセスごとのベスト値を全プロセス間で共有してグローバルベスト値を求める
        global_best_value_g = comm.allreduce(global_best_value, op=MPI.MIN)
        if global_best_value == global_best_value_g : # 自分のプロセスのベスト値がグローバルベスト値であったとき
          rank_l = rank
          global_best_position_l = global_best_position
          global_best_index_l    = global_best_index_hisotry[i]
        else:
          rank_l = -1
          global_best_position_l = global_best_position
          global_best_index_l    = global_best_index_hisotry[i]
        rank_g = comm.allreduce(rank_l, op=MPI.MAX) # グローバルベスト値を有するプロセス(ID: rank)がどれであるか共有する
        global_best_position   = comm.bcast(global_best_position_l, root=rank_g) # グローバルベスト値をとる条件値を全プロセスに送信する
        global_best_value      = global_best_value_g
        #global_best_index_g    = comm.bcast(global_best_index_l, root=rank_g)
        global_best_index_hisotry[i] = comm.bcast(global_best_index_l, root=rank_g)

      # 粒子位置と速度の更新
      for n in range(start, end):
        # History
        particle_velocity_viz[i,n,:] =  particle_velocity[n][:]
        particle_position_viz[i,n,:] =  particle_position[n][:]
        # Update position and velocity of particle for next step
        rand1 = np.random.rand(dim)
        rand2 = np.random.rand(dim)
        cognitive_velocity = cognitive_coef * rand1 * (particle_best_position[n] - particle_position[n])
        social_velocity = social_coef * rand2 * (global_best_position - particle_position[n])
        particle_velocity[n] = inertia * particle_velocity[n] + cognitive_velocity + social_velocity
        particle_position[n] = particle_position[n] + particle_velocity[n]      

      # Display status
      if comm.rank == 0:
        print('Ste:p',i, ' Best solution:',global_best_value, ' Parameter:',global_best_position)


    # MPI process for history data
    if flag_mpi :
      particle_position_viz = comm.allreduce(particle_position_viz, op=MPI.SUM)
      particle_velocity_viz = comm.allreduce(particle_velocity_viz, op=MPI.SUM)
      particle_solution     = comm.allreduce(particle_solution, op=MPI.SUM)
      particle_rank[start:end] = rank
      particle_rank = comm.allreduce(particle_rank, op=MPI.SUM)


    # Tecplot format
    if comm.rank == 0:
      filename_tmp='tecplot_particle.dat'
      file_output = open( filename_tmp , 'w')

      header_tmp = "Variables = X, Y, U, V, ID, Rank, Solution"  + '\n'
      file_output.write( header_tmp )

      for i in range(0, max_iter):
        text_tmp = 'zone t="Time'+str(i+1) +' sec"' + '\n'
        text_tmp =  text_tmp + 'i='+str(num_particles)+' f=point' + '\n'
        for n in range(0, num_particles):
          text_tmp = text_tmp
          for m in range(0,dim):
            text_tmp = text_tmp  + str( particle_position_viz[i,n,m] ) + ', '
          for m in range(0,dim):
            text_tmp = text_tmp  + str( particle_velocity_viz[i,n,m] ) + ', '
          solution_tmp = str(particle_solution[i,n])
          text_tmp = text_tmp + str(n+1) + ', ' + str(particle_rank[n]) + ', ' + str(solution_tmp) + '\n'
        file_output.write( text_tmp )

      file_output.close()

    return global_best_position, global_best_value


if __name__ == "__main__":

  # MPI settings
  mpi_instance = mpi()

  # PSO
  optimizer_pso = optimizer_pso()
  best_position, best_value = optimizer_pso.pso(mpi_instance)

  if mpi_instance.rank == 0:
    print("Best position:", best_position)
    print("Best value:", best_value)

