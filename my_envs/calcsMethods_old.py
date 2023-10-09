import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict
import cProfile  as cProfile


#region Temperature of soil
def temp_soil(dt, T_soil_0, R=0.35, h=0.35, T_air=299, _lambda=np.random.uniform(0.8, 1.16)):
    def heat_equation(t, T):
        r = np.linspace(1e-6, R, 100)  # Избегаем нулевых значений r
        z = np.linspace(1e-6, h, 100)  # Избегаем нулевых значений z
        # Коэффициенты теплопроводности
        alpha_r = _lambda**2
        alpha_z = _lambda**2
        # Уравнение теплопроводности
        dT_dr = alpha_r * (1 / r) * np.gradient(r * np.gradient(T.reshape((100, 100)), axis=1), axis=1)
        dT_dz = alpha_z * (1 / z) * np.gradient(np.gradient(T.reshape((100, 100)), axis=0), axis=0)
        # Теплообмен с окружающей средой (воздухом)
        heat_exchange = (T_air - T) * 0.01  # Простая модель теплообмена
        # Сбор результатов в одномерный массив
        dT_dt = (dT_dr + dT_dz).flatten() + heat_exchange.flatten()
        return dT_dt

    t_span = (0, dt)
    T_initial = np.ones((100, 100)) * T_soil_0  # Начальное распределение температуры

    sol = solve_ivp(heat_equation, t_span, T_initial.flatten(), t_eval=[dt])
    return np.clip(sol.y.mean(),0,40)
#endregion
#region Humidity of soil
def e_s(T_air:float) -> float:
    """
    Насыщенное парциальное давление воды при данной температуре

    Args:
    T_air (float): температура в градусах Цельсия

    Returns:
    float: Насыщенное парциальное давление воды при данной температуре

    """
    return 0.6108 * np.exp(17.27*T_air / (T_air + 237.3))
def e_a(RH:float,e_s:float) -> float:
    """
    Формула для расчета фактического парциального давления воды e_a 
    при данной температуре и относительной влажности RH
    Args:
        RH (float): относительной влажности
        e_s (float): Насыщенное парциальное давление воды при данной температуре

    Returns:
        float:  фактическое парциальное давление воды
    """
    return RH * e_s

def calculate_diff_eq(dt,
                      RH, AH,  
                      T_air,#e_s, e_a,
                      k=np.random.uniform(0.5,2),
                      K=np.random.uniform(0.1,10),  
                      c_p=1005, 
                      lmbda=2.45*10**6, 
                      R_in=150, R_out=75, # Вт/м²
                      A=769366.9):
    _e_s = e_s(T_air)
    _e_a = e_a(RH, _e_s)
    diff_eq_result = dt * (k * (RH - AH) * A - (K * (_e_s - _e_a) + (c_p * (R_in - R_out)) / lmbda))

    # Используйте np.clip для предотвращения переполнения
    return np.clip(diff_eq_result, -1e10, 1e10)
#endregion
#region pH level of soil
def pH(pH_solution, pH_soil, dt):
    """
    Рассчитывает изменение pH почвы за промежуток времени dt.

    Args:
        pH_solution (float): pH в подведенном растворе.
        pH_soil (float): Текущее значение pH в почве.
        dt (float): Промежуток времени (в часах).

    Returns:
        float: Изменение pH почвы за промежуток времени dt.
    """
    delta_pH = (pH_solution - pH_soil) * dt
    return delta_pH
#endregion
#region EC/TDS of soil
def calculate_delta_EC_TDS(EC_TDS_solution, EC_TDS_soil, k_assim_EC_TDS, dt):
    """
    Рассчитывает изменение EC/TDS в почве за промежуток времени dt.

    Args:
        EC_TDS_solution (float): Электропроводность (или TDS) в подведенном растворе (mS/cm).
        EC_TDS_soil (float): Текущее значение электропроводности (или TDS) в почве (mS/cm).
        k_assim_EC_TDS (float): Коэффициент ассимиляции для EC/TDS.
        dt (float): Промежуток времени (в часах).

    Returns:
        float: Изменение EC/TDS в почве за промежуток времени dt.
    """
    delta_EC_TDS = (EC_TDS_solution - EC_TDS_soil) * k_assim_EC_TDS * dt
    return delta_EC_TDS
#endregion
#region N,P,K of soil
def calculate_concentration_change(C_soil, C_solution, k_assim, dt):
    """
    Расчет изменения концентрации элемента в почве

    Args:
    C_soil (float): Текущая концентрация элемента в почве (ppm)
    C_solution (float): Концентрация элемента в подведенном растворе (ppm)
    k_assim (float): Коэффициент ассимиляции для элемента
    dt (float): Промежуток времени (часы)

    Returns:
    float: Изменение концентрации элемента в почве за промежуток времени
    """
    delta_C = (C_solution - C_soil) * k_assim * dt
    return delta_C
#endregion
#region Utils
def V_i(dt,a_i:float=0.5,U:float=15):
    return np.clip(a_i, 0, 1) * dt *U
#endregion

#region General
#region old step_i_Soil
def _step_i_Soil(
    valve_growth,
    valve_fruiting,
    valve_water,
    vel_assim,
    low_high_dict:Dict[str,np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    num_iter: int,
    start_time: float,
    time_step: int,
    Volume:float=8.5, # объем горшка в мм3,
                flow_vel=15, # л/час
                flow_relief=50 ,# л/час
    isProfile:bool=False
):
    
    if isProfile:
        profiler = cProfile.Profile()  # Создаем объект профилирования
        profiler.enable()  # Включаем профилирование
        
        print('START "_step_i_Soil" profile')
    weight_charge = {
            'N':np.array([0.4 , -3]),# N [ppm]
            'P':np.array([0.3 , -3]),# P [ppm]
            'K':np.array([0.2 , -1]),# K [ppm]
            'Mg':np.array([0.05 , 2]),# P [ppm]
            'Ca':np.array([0.05 , 2]),# K [ppm]
    }
    k_assim = {
            'N':np.array([0.5 , 2]),# N [ppm]
            'P':np.array([0.01 , 0.1]),# P [ppm]
            'K':np.array([0.5 , 2]),# K [ppm]
            'Mg':np.array([0.01 , 1]),# P [ppm]
            'Ca':np.array([0.1 , 0.5]),# K [ppm]
            'EC':np.array([0.01,2.0])
    }
    end_time = start_time
    state_params = dict(zip(list(low_high_dict.keys()), state.tolist()))
    dtime = time_step / num_iter
    [vel_growth, vel_fruiting, vel_water], vel_relief, T_air, phi_air = [V_i(dt=dtime, a_i=a_i, U=flow_vel) for a_i in action[:3]], V_i(dt=dtime, a_i=action[3],U=flow_relief), 16 + 14 * action[4], action[-1]
    N1_gen, N2_gen, N3_gen, N_relief_gen = 0, 0, 0, 0
    for i in range(num_iter):
        # region 0. Calcs newmixture parameters
        N1, N2, N3, N_relief = [vel * dtime for vel in [vel_growth, vel_fruiting, vel_water, vel_relief]]
        # vel_assim = np.mean(np.array([state_params[f'{elem}_mix'] * np.random.uniform(k_assim[elem][0], k_assim[elem][1]) * A/3600*1e-6 for elem in ['N', 'P', 'K', 'Mg', 'Ca']]))
        N_relief += np.random.uniform(vel_assim[0],vel_assim[1]) * dtime 
        for elem in ['N', 'P', 'K', 'Mg', 'Ca', 'pH']:
            state_params[f'{elem}_mix'] += (sum([valve[elem] * N for N, valve in
                                                  zip([N1, N2, N3], [valve_growth, valve_fruiting, valve_water])]) - N_relief * state_params[
                                                f'{elem}_mix']) / \
                                           (sum([N1, N2, N3]) - N_relief)
            # Apply parameter value constraints
            state_params[f'{elem}_mix'] = np.clip(state_params[f'{elem}_mix'], low_high_dict[f'{elem}_mix'][0], low_high_dict[f'{elem}_mix'][1])
            
       # Обновление EC_mix с условием, чтобы значение не становилось отрицательным
        state_params['EC_mix'] = (Volume*state_params['EC_mix'] + N1 * valve_growth['EC'] + N2 * valve_fruiting['EC'] + N3 * valve_water['EC']  - N_relief * state_params['EC_mix'])/(Volume  + N1 + N2 + N3 - N_relief )
        state_params['EC_mix'] = np.clip(state_params[f'EC_mix'], low_high_dict[f'EC_mix'][0], low_high_dict[f'EC_mix'][1])
            
        # endregion
        # region 1. Calcs new T_soil
        state_params['T_soil'] = temp_soil(dt=dtime, T_soil_0=state_params['T_soil'], T_air=T_air)
        # endregion
        # region 2. Calcs new phi_soil
        state_params['phi_soil'] += calculate_diff_eq(
            dt=dtime,
            RH=state_params['phi_soil'],
            AH=phi_air, T_air=T_air
        )
        state_params['phi_soil'] = np.clip(state_params['phi_soil'],low_high_dict['phi_soil'][0], low_high_dict['phi_soil'][1])
        # endregion
        # region 3. Calc new pH_soil
        state_params['pH_soil'] += pH(pH_solution=state_params['pH_mix'], 
                                      pH_soil=state_params['pH_soil'], dt=dtime)
        state_params['pH_soil'] = np.clip(state_params['pH_soil'],low_high_dict['pH_soil'][0], low_high_dict['pH_soil'][1])
        # endregion
        # region 4. Calc new EC/TDS
        state_params['EC_soil'] += calculate_delta_EC_TDS(
            EC_TDS_solution=state_params['EC_mix'],
            EC_TDS_soil=state_params['EC_soil'],
            k_assim_EC_TDS=np.random.uniform(k_assim['EC'][0],k_assim['EC'][1]),dt=dtime
            )
        
        state_params['EC_soil']= np.clip(state_params['EC_soil'],low_high_dict['EC_soil'][0], low_high_dict['EC_soil'][1])
        # endregion
        # region 5. Calcs N,P,K,Mg,Ca
        for elem in ['N', 'P', 'K', 'Mg', 'Ca']:
            state_params[f'{elem}_soil'] += calculate_concentration_change(
                C_soil=state_params[f'{elem}_soil'],C_solution=state_params[f'{elem}_mix'],
                k_assim=np.random.uniform(k_assim[elem][0],k_assim[elem][1]),dt=dtime)
            
            state_params[f'{elem}_soil'] = np.clip(state_params[f'{elem}_soil'],low_high_dict[f'{elem}_soil'][0], low_high_dict[f'{elem}_soil'][1])
        #endregion 
        N1_gen += N1
        N2_gen += N2
        N3_gen += N3
        N_relief_gen += N_relief
        end_time += dtime
    if isProfile:
        profiler.disable()  # Отключаем профилирование
        profiler.print_stats(sort='cumulative')
        
        print('END "_step_i_Soil" profile')
    return state_params, end_time, Volume + N1_gen + N2_gen + N3_gen - N_relief_gen 
#endregion
#endregion
