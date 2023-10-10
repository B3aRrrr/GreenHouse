__credits__ = ["Dmitry CHERNYSHEV"]

from .calcsMethods import _step_i_Soil
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from typing import List

import gym
from gym.utils import EzPickle
import cProfile  as cProfile  # Импортируем cProfile



'''
Этап семенного посева
    Температура почвы: [20°C, 25°C]
    Влажность почвы: [50%, 60%]
    pH кислотность почвы: [6.0, 6.5]
    EC/TDS почвы: [0.2 mS/cm, 0.5 mS/cm]
    N в почве: [100 ppm, 200 ppm]
    P в почве: [50 ppm, 100 ppm]
    K в почве: [150 ppm, 300 ppm]

Этап рассады
    Температура почвы: [18°C, 24°C]
    Влажность почвы: [60%, 70%]
    pH кислотность почвы: [6.0, 6.5]
    EC/TDS почвы: [0.2 mS/cm, 0.8 mS/cm]
    N в почве: [100 ppm, 200 ppm]
    P в почве: [50 ppm, 100 ppm]
    K в почве: [150 ppm, 300 ppm]

Этап цветения и плодоношения
    Температура почвы: [21°C, 24°C]
    Влажность почвы: [70%, 80%]
    pH кислотность почвы: [6.0, 6.5]
    EC/TDS почвы: [1.0 mS/cm, 2.0 mS/cm]
    N в почве: [100 ppm, 200 ppm]
    P в почве: [50 ppm, 100 ppm]
    K в почве: [150 ppm, 300 ppm]

Этап созревания плодов
    Температура почвы: [20°C, 23°C]
    Влажность почвы: [60%, 70%]
    pH кислотность почвы: [6.0, 6.5]
    EC/TDS почвы: [0.8 mS/cm, 1.5 mS/cm]
    N в почве: [100 ppm, 200 ppm]
    P в почве: [50 ppm, 100 ppm]
    K в почве: [150 ppm, 300 ppm]
    
'''


class SoilGreenHouse(gym.Env, EzPickle):
    #region valves vars
    mixtures_growth = {
        0:{
            'N':20/100*10**4,
            'P':10/100*10**4,
            'K':20/100*10**4,
            'Mg':5/100*10**4,
            'Ca':2/100*10**4,
            'pH':6.0,
            'EC':1.2
        },
        1:{
            'N':12/100 *10**4,
            'P':8/100*10**4,
            'K':10/100*10*4,
            'Mg':3/100*10**4,
            'Ca':3/100*10**4,
            'pH':6.5,
            'EC':1.3
        },
        2:{
            'N':0.025/100*10**4,
            'P':0.005/100*10**4,
            'K':0.015/100*10**4,
            'Mg':0.003/100*10**4,
            'Ca':0.003/100*10**4,
            'pH':6.2,
            'EC':1.4
        },
    }
    mixtures_fruiting = {
        0:{        
            'N':10/100*10**4,
            'P':15/100*10**4,
            'K':30/100*10**4,
            'Mg':5/100*10**4,
            'Ca':5/100*10**4,
            'pH':6.5,
            'EC':1.8
        },
        1:{
            'N':25/100*10**4,
            'P':5/100*10**4,
            'K':15/100*10**4,
            'Mg':3/100*10**4,
            'Ca':3/100*10**4,
            'pH':6.2,
            'EC':1.4
        },
        2:{
            'N':15/100*10**4,
            'P':10/100*10**4,
            'K':25/100*10**4,
            'Mg':5/100*10**4,
            'Ca':5/100*10**4,
            'pH':6.3,
            'EC':1.5
        },
    }
    water={
        'N':0,
        'P':0,
        'K':0,
        'Mg':0,
        'Ca':0,
        'pH':7,
        'EC':1.8
    },
    #endregion
    #region stages
    
# Стадия посева семян (seed sowing): 0.2 - 0.5 л/час
# Стадия рассады (seedling): 0.5 - 2.0 л/час
# Стадия цветения и плодоношения (flowering_fruiting): 1.0 - 3.0 л/час
# Стадия созревания плодов (fruit ripening): 1.5 - 4.0 л/час
    stages = {
        'seed_sowing':{
                'T_soil':np.array([20,25]),# T [°C]
                'phi_soil':np.array([.5,.6]),# phi [%]
                'pH_soil':np.array([6.0,6.5]),# pH
                'EC_soil':np.array([0.2 , 0.5]),# EC/TDS [mS/cm]
                'N_soil':np.array([100 , 200]),# N [ppm]
                'P_soilP':np.array([50 , 100]),# P [ppm]
                'K_soil':np.array([150 , 300]),# K [ppm]
                'vel_assim':np.array([0.2,0.5]) # л/час
           },
        'seedling':{
                'T_soil':np.array([18,24]),# T [°C]
                'phi_soil':np.array([.6,.7]),# phi [%]
                'pH_soil':np.array([6.0,6.5]),# pH
                'EC_soil':np.array([0.2 , 0.8]),# EC/TDS [ mS/cm]
                'N_soil':np.array([100 , 200]),# N [ppm]
                'P_soil':np.array([50 , 100]),# P [ppm]
                'K_soil':np.array([150 , 300]),# K [ppm]
                'vel_assim':np.array([0.5,2.0]) # л/час
            },
        'flowering_fruiting':{
                'T_soil':np.array([21,24]),# T [°C]
                'phi_soil':np.array([.7,.8]),# phi [%]
                'pH_soil':np.array([6.0,6.5]),# pH
                'EC_soil':np.array([1.0 , 2.0]),# EC/TDS [ mS/cm]
                'N_soil':np.array([100 , 200]),# N [ppm]
                'P_soil':np.array([50 , 100]),# P [ppm]
                'K_soil':np.array([150 , 300]),# K [ppm]
                'vel_assim':np.array([1,3]) # л/час
            },
        'fruit_ripening':{
                'T_soil':np.array([20,23]),# T [°C]
                'phi_soil':np.array([.6,.7]),# phi [%]
                'pH_soil':np.array([6.0,6.5]),# pH
                'EC_soil':np.array([0.8 , 1.5]),# EC/TDS [ mS/cm]
                'N_soil':np.array([100 , 200]),# N [ppm]
                'P_soil':np.array([50 , 100]),# P [ppm]
                'K_soil':np.array([150 , 300]),# K [ppm]
                'vel_assim':np.array([1.5,4.0]) # л/час
            }
    }
    #endregion
    """
    ### Description
    This is a simple 4-joint walker robot environment.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gym/envs/box2d/bipedal_walker.py
    ```

    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.

    ### Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ### Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ### Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ### Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ### Arguments
    To use to the _hardcore_ environment, you need to specify the
    `hardcore=True` argument like below:
    ```python
    import gym
    env = gym.make("BipedalWalker-v3", hardcore=True)
    ```

    ### Version History
    - v3: returns closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ### References -->

    ### Credits
    Created by Oleg Klimov

    """  
    def __init__(self, 
                 stage:str,
                 valves_coefs:list,
                Volume:float=8.5, # объем горшка в мм3,
                flow_vel=15, # л/час
                flow_relief=50, # л/час
                
                random_state:int=42,
                eps:float=1e-5,
                render_mode:bool=False,
                time_step:int=1, # час
                num_iter:int=100,
                A=769366.9, # площадь корней в мм2
                obs_params_names:List[str] = [
        'T_soil','phi_soil','pH_soil','EC_soil','N_soil','P_soil','K_soil','Mg_soil','Ca_soil',
        'pH_mix','EC_mix','N_mix','P_mix','K_mix','Mg_mix','Ca_mix',
        ],
                act_params_names:List[str] = ['valve_growth','valve_gfruiting','valve_water','valve_relief','valve_T_air','valve_phi_air',]
                 ):
        EzPickle.__init__(self, stage, A,valves_coefs,Volume,random_state,eps,render_mode,flow_relief,num_iter,obs_params_names,act_params_names)
        
        self._max_episode_steps = 72  # Установите максимальное количество шагов в эпизоде
        
        self.Volume =Volume
        '''
        PARAMETERS:
            Температура почвы в горшке:
                Минимальная: Около [0,40]°C 
            Влажность почвы в горшке:
                Минимальная: Около [5,100]% 
            pH кислотность почвы:
                Минимальная: Около pH [3,9]
            EC/TDS почвы в среде:
                Минимальная: Около [100,3000] μS/cm
            Азот (N):
                Минимальное : Около [10,5000] ppm (чрезмерное удобрение азотом может привести к загрязнению почвы и водных ресурсов).
            Фосфор (P):
                Минимальное значение: Около [2,1000] ppm (высокие концентрации фосфора могут вызвать экологические проблемы и стоить дорого в производстве).
            Калий (K):
                Минимальное значение: Около [50,1000] ppm (чрезмерное удобрение калием может нарушить баланс других элементов и привести к недостатку магния или кальция).
            Температура воздушной среды:
                Минимальная: Около [-40,50]°C и выше (в жаркое летнее время).
            Влажность воздушной среды:
                Минимальная: Около [10,100]% (при насыщении воздуха водяным паром).
        '''
        low = np.array(
            [
                # SOIL
                0, # T_soil [°C]
                .5, # phi_soil [%]
                3, # pH
                100,# EC/TDS [μS/cm]
                10, # N [ppm]
                2, # P [ppm]
                50, # K [ppm]
                10, # Mg [ppm]
                100, # Ca [ppm]
                # AIR
                # 12, # # T_air [°C]
                # .1, # phi_air [%]
                # MIXTURE
               5, # pH
                0,# EC/TDS [μS/cm]
                0, # N [ppm]
                0, # P [ppm]
                0, # K [ppm]
                0, # Mg [ppm]
                0, # Ca [ppm]
        ]).astype(np.float32)
        high = np.array(
            [
                # SOIL
                40, # # T_soil [°C]
                1, # phi_soil [%]
                9, # pH
                3000,# EC/TDS [μS/cm]
                5000, # N [ppm]
                1000, # P [ppm]
                1000, # K [ppm]
                3000, # Mg [ppm]
                10000, # Ca [ppm]
                # AIR
                # 32, # # T_air [°C]
                # 1.0, # phi_air [%]
                # MIXTURE
                9, # pH
                3000,# EC/TDS [μS/cm]
                50_000, # N [ppm]
                25_000, # P [ppm]
                50_000, # K [ppm]
                10_000, # Mg [ppm]
                10_000, # Ca [ppm]
            ]).astype(np.float32)
        assert len(low) == len(obs_params_names)
        self.act_params_names = act_params_names
        '''
        1)Vent_1: Питательная смесь "Ростовая" [0,1]:

            Состав (на 100 литров воды):
                Азот (N): 20 г
                Фосфор (P): 10 г
                Калий (K): 20 г
                Магний (Mg): 5 г
                Кальций (Ca): 5 г
                Кислотность (pH): 6.0
                EC/TDS: 1.2 mS/cm
            
        2)Vent_2: Питательная смесь "Цветение и плодоношение":
        
            Состав (на 100 литров воды):
                Азот (N): 10 г
                Фосфор (P): 15 г
                Калий (K): 30 г
                Магний (Mg): 5 г
                Кальций (Ca): 5 г
                Кислотность (pH): 6.5
                EC/TDS: 1.8 mS/cm
            
        3)Vent_3: Чистая вода:
            Состав (на 100 литров воды):
                Азот (N): 0 г
                Фосфор (P): 0 г
                Калий (K): 0 г
                Магний (Mg): 0 г
                Кальций (Ca): 0 г
                Кислотность (pH): 7.0
                EC/TDS: 0 mS/cm
        
            
        '''

        self.action_space = gym.spaces.Box(
            np.zeros(len(self.act_params_names),dtype=np.float32),
            np.ones(len(self.act_params_names),dtype=np.float32), 
        )
        
        
        self.observation_space = gym.spaces.Box(low, high)
        self.low_high_dict = dict(zip(obs_params_names,np.concatenate((low[:,np.newaxis], high[:,np.newaxis]), axis=1)))
        self.optimal_space = self.__class__.stages[stage]
        # VALVES (actions [0,1])
        self.valve_growth = self.__class__.mixtures_growth[valves_coefs[0]]
        self.valve_fruiting = self.__class__.mixtures_fruiting[valves_coefs[1]]
        self.valve_water = self.__class__.water[0]
        
        self.time_step= time_step
        self.flow_vel = flow_vel
        self.flow_relief = flow_relief
        self.current_time=0
        
        self.state  = self.reset()
        self.num_iter = num_iter
        self.A=A     
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed) 
        self.current_time=0
        return self.observation_space.sample()       
    
    def step(self, action: np.ndarray,isProfile:False):
        if isProfile:
            profiler = cProfile.Profile()  # Создаем объект профилирования
            profiler.enable()  # Включаем профилирование
            print('START "step" profile')

        new_state_params, self.current_time, dV = _step_i_Soil(
            valve_growth=self.valve_growth,
            valve_fruiting=self.valve_fruiting,
            valve_water=self.valve_water,
            vel_assim=self.optimal_space['vel_assim'],
            low_high_dict=self.low_high_dict,
            state=self.state,
            action=action,
            num_iter=self.num_iter,
            start_time=self.current_time,
            time_step=self.time_step,
            Volume=self.Volume,
            flow_vel=15,
            flow_relief=50,isProfile=isProfile
        )
        
        if isProfile:
            profiler.disable()  # Отключаем профилирование
            profiler.print_stats(sort='cumulative')
            
            print('END "step" profile')


        self.state = np.array(list(new_state_params.values()))
        reward = self._get_reward(new_state_params, dV)
        info = {}
        done = False
        if  self.current_time >= self._max_episode_steps:
                done = True
        return self.state, reward, done, info  
    
    def _get_reward(self, state_params,dV):     
      keys = self.optimal_space.keys()
      def calc_loc_rew(state_params,param_name,best_r,N_r):
          return np.clip(np.exp(-(state_params[param_name] - np.mean(self.optimal_space[param_name]))**2),-1,1)*best_r/N_r
      reward_list = [calc_loc_rew(state_params,key,300,N_r=len(keys)) for key in keys if key != 'vel_assim']
      reward_list.append(np.clip(np.exp(-(dV)**2),-1,1)*300/(len(keys)))
      clipped_reward = np.clip(np.sum(reward_list), -300, 300)
      
      return clipped_reward
    def render(self, mode='human'):
        # Получите значения параметров среды (замените этот код на свой)
        parameters = self.get_parameters()

        # Определите количество параметров (N)
        N = len(parameters)

        # Создайте подграфики для каждого параметра
        fig, axs = plt.subplots(N, figsize=(6, 4 * N))

        # Для каждого параметра создайте гистограмму
        for i in range(N):
            ax = axs[i]
            parameter_values = parameters[i]

            # Рассчитайте гистограмму
            hist, bins = np.histogram(parameter_values, bins=20)  # Можете настроить количество бинов

            # Отобразите гистограмму
            ax.hist(parameter_values, bins, alpha=0.5, color='blue')
            ax.set_title(f'Parameter {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        # Регулируйте расположение графиков и т. д. по своему усмотрению

        # Покажите графики
        plt.tight_layout()
        plt.show()
