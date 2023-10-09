__credits__ = ["Dmitry CHERNYSHEV"]

<<<<<<< HEAD
from .calcsMethods import _step_i_Soil
=======
>>>>>>> 7801a6405ab59690a91b1a87582234b159f218ee
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
<<<<<<< HEAD
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
=======

import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.hull == contact.fixtureA.body
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalker(gym.Env, EzPickle):
>>>>>>> 7801a6405ab59690a91b1a87582234b159f218ee
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

<<<<<<< HEAD
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
        
        self.state, _ = self.reset()
        self.num_iter = num_iter
        self.A=A     
        
=======
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        EzPickle.__init__(self, render_mode, hardcore)
        self.isopen = True

        self.world = Box2D.b2World()
        self.terrain: List[Box2D.b2Body] = []
        self.hull: Optional[Box2D.b2Body] = None

        self.prev_shaping = None

        self.hardcore = hardcore

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -math.pi,
                -5.0,
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
            ]
            + [-1.0] * 10
        ).astype(np.float32)
        high = np.array(
            [
                math.pi,
                5.0,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
            ]
            + [1.0] * 10
        ).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        # state = [
        #     self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
        #     2.0 * self.hull.angularVelocity / FPS,
        #     0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
        #     0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        #     self.joints[
        #         0
        #     ].angle,  # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
        #     self.joints[0].speed / SPEED_HIP,
        #     self.joints[1].angle + 1.0,
        #     self.joints[1].speed / SPEED_KNEE,
        #     1.0 if self.legs[1].ground_contact else 0.0,
        #     self.joints[2].angle,
        #     self.joints[2].speed / SPEED_HIP,
        #     self.joints[3].angle + 1.0,
        #     self.joints[3].speed / SPEED_KNEE,
        #     1.0 if self.legs[3].ground_contact else 0.0,
        # ]
        # state += [l.fraction for l in self.lidar]

        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        original_y = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.random() > 0.5 else -1
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

>>>>>>> 7801a6405ab59690a91b1a87582234b159f218ee
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
<<<<<<< HEAD
        super().reset(seed=seed) 
        self.current_time=0
        return self.observation_space.sample(), {}        
    
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

        return self.state, reward, False, info  
    
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
=======
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs: List[Box2D.b2Body] = []
        self.joints: List[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class BipedalWalkerHardcore:
    def __init__(self):
        raise error.Error(
            "Error initializing BipedalWalkerHardcore Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
            'gym.make("BipedalWalker-v3", hardcore=True)'
        )


if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalker()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        if terminated or truncated:
            break
>>>>>>> 7801a6405ab59690a91b1a87582234b159f218ee
