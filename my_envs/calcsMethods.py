import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
from typing import Dict
import cProfile as cProfile

GRID_SIZE = 100
ZERO_PADDING = 1e-6
THERMAL_CONDUCTIVITY = 1.0
HEAT_EXCHANGE_COEFF = 0.1
k_assim = {
        'N': np.array([0.5, 2]),
        'P': np.array([0.01, 0.1]),
        'K': np.array([0.5, 2]),
        'Mg': np.array([0.01, 1]),
        'Ca': np.array([0.1, 0.5]),
        'EC': np.array([0.01, 2.0])
    }

# @jit(nopython=True)
def calculate_gradient(arr, axis):
    diff = np.diff(arr, axis=axis)
    gradient = np.zeros_like(arr)
    
    if axis == 0:
        gradient[:-1, :] = diff
        gradient[-1, :] = arr[-1, :] - arr[-2, :]
    elif axis == 1:
        gradient[:, :-1] = diff
        gradient[:, -1] = arr[:, -1] - arr[:, -2]
    
    return gradient

# @jit(nopython=True)
def heat_equation(t, T, r, z, alpha_r, alpha_z, T_air):
    
    dT_dr = alpha_r * (1 / r) * calculate_gradient(r * calculate_gradient(T.reshape((GRID_SIZE, GRID_SIZE)), axis=1), axis=1)
    dT_dz = alpha_z * (1 / z) * calculate_gradient(calculate_gradient(T.reshape((GRID_SIZE, GRID_SIZE)), axis=0), axis=0)
    heat_exchange = (T_air - T) * HEAT_EXCHANGE_COEFF
    dT_dt = (dT_dr + dT_dz).flatten() + heat_exchange.flatten()
    return dT_dt


def temp_soil(dt, T_soil_0, R=0.35, h=0.35, T_air=299, _lambda=None):
    if _lambda is None:
        _lambda = THERMAL_CONDUCTIVITY

    r = np.linspace(ZERO_PADDING, R, GRID_SIZE)
    z = np.linspace(ZERO_PADDING, h, GRID_SIZE)

    alpha_r = _lambda ** 2
    alpha_z = _lambda ** 2

    t_span = (0, dt)
    T_initial = np.ones((GRID_SIZE, GRID_SIZE)) * T_soil_0

    sol = solve_ivp(
        lambda t, T: heat_equation(t, T, r, z, alpha_r, alpha_z, T_air),
        t_span,
        T_initial.flatten(),
        t_eval=[dt]
    )

    return np.clip(sol.y.mean(), 0, 40)


def calculate_diff_eq(dt, RH, AH, T_air, k=None, K=None, c_p=1005, lmbda=2.45e6, R_in=150, R_out=75, A=769366.9):
    if k is None:
        k = np.random.uniform(0.5, 2)
    if K is None:
        K = np.random.uniform(0.1, 10)

    _e_s = e_s(T_air)
    _e_a = e_a(RH, _e_s)
    diff_eq_result = dt * (k * (RH - AH) * A - (K * (_e_s - _e_a) + (c_p * (R_in - R_out)) / lmbda))

    return np.clip(diff_eq_result, -1e10, 1e10)

@jit(nopython=True)
def e_s(T_air: float) -> float:
    return 0.6108 * np.exp(17.27 * T_air / (T_air + 237.3))

@jit(nopython=True)
def e_a(RH: float, e_s: float) -> float:
    return RH * e_s


def calculate_diff_eq(dt, RH, AH, T_air, k=None, K=None, c_p=1005, lmbda=2.45e6, R_in=150, R_out=75, A=769366.9):
    if k is None:
        k = np.random.uniform(0.5, 2)
    if K is None:
        K = np.random.uniform(0.1, 10)

    _e_s = e_s(T_air)
    _e_a = e_a(RH, _e_s)
    diff_eq_result = dt * (k * (RH - AH) * A - (K * (_e_s - _e_a) + (c_p * (R_in - R_out)) / lmbda))

    return np.clip(diff_eq_result, -1e10, 1e10)

@jit(nopython=True)
def pH(pH_solution, pH_soil, dt):
    delta_pH = (pH_solution - pH_soil) * dt
    return delta_pH

@jit(nopython=True)
def calculate_delta_EC_TDS(EC_TDS_solution, EC_TDS_soil, k_assim_EC_TDS, dt):
    delta_EC_TDS = (EC_TDS_solution - EC_TDS_soil) * k_assim_EC_TDS * dt
    return delta_EC_TDS

@jit(nopython=True)
def calculate_concentration_change(C_soil, C_solution, k_assim, dt):
    delta_C = (C_solution - C_soil) * k_assim * dt
    return delta_C

@jit(nopython=True)
def V_i(dt, a_i: float = 0.5, U: float = 15):
    return a_i * dt * U

def _step_i_Soil(
    valve_growth,
    valve_fruiting,
    valve_water,
    vel_assim,
    low_high_dict: Dict[str, np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    start_time: float,
    time_step: int,
    Volume: float = 8.5,
    flow_vel=15,
    flow_relief=50,
    isProfile: bool = False
):
    if isProfile:
        profiler = cProfile.Profile()
        profiler.enable()
            
        print('START "_step_i_Soil" profile')

    end_time = start_time + time_step
    state_params = dict(zip(list(low_high_dict.keys()), state.tolist()))
    dtime = time_step

    [vel_growth, vel_fruiting, vel_water], vel_relief, T_air, phi_air = [V_i(dt=dtime, a_i=a_i, U=flow_vel) for a_i in action[:3]], V_i(dt=dtime, a_i=action[3], U=flow_relief), 16 + 14 * action[4], action[-1]

    
    N1, N2, N3, N_relief = [vel * dtime for vel in [vel_growth, vel_fruiting, vel_water, vel_relief]]
    N_relief += np.random.uniform(vel_assim[0], vel_assim[1]) * dtime

    for elem in ['N', 'P', 'K', 'Mg', 'Ca', 'pH']:
        state_params[f'{elem}_mix'] += (sum([valve[elem] * N for N, valve in zip([N1, N2, N3], [valve_growth, valve_fruiting, valve_water])]) - N_relief * state_params[f'{elem}_mix']) / (sum([N1, N2, N3]) - N_relief)
        state_params[f'{elem}_mix'] = np.clip(state_params[f'{elem}_mix'], low_high_dict[f'{elem}_mix'][0], low_high_dict[f'{elem}_mix'][1])

    state_params['EC_mix'] = (Volume * state_params['EC_mix'] + N1 * valve_growth['EC'] + N2 * valve_fruiting['EC'] + N3 * valve_water['EC'] - N_relief * state_params['EC_mix']) / (Volume + N1 + N2 + N3 - N_relief)
    state_params['EC_mix'] = np.clip(state_params[f'EC_mix'], low_high_dict[f'EC_mix'][0], low_high_dict[f'EC_mix'][1])

    state_params['T_soil'] = temp_soil(dt=dtime, T_soil_0=state_params['T_soil'], T_air=T_air)

    state_params['phi_soil'] += calculate_diff_eq(
        dt=dtime,
        RH=state_params['phi_soil'],
        AH=phi_air, T_air=T_air
    )
    state_params['phi_soil'] = np.clip(state_params['phi_soil'], low_high_dict['phi_soil'][0], low_high_dict['phi_soil'][1])

    state_params['pH_soil'] += pH(pH_solution=state_params['pH_mix'],
                                pH_soil=state_params['pH_soil'], dt=dtime)
    state_params['pH_soil'] = np.clip(state_params['pH_soil'], low_high_dict['pH_soil'][0], low_high_dict['pH_soil'][1])

    state_params['EC_soil'] += calculate_delta_EC_TDS(
        EC_TDS_solution=state_params['EC_mix'],
        EC_TDS_soil=state_params['EC_soil'],
        k_assim_EC_TDS=np.random.uniform(k_assim['EC'][0], k_assim['EC'][1]), dt=dtime
    )

    state_params['EC_soil'] = np.clip(state_params['EC_soil'], low_high_dict['EC_soil'][0], low_high_dict['EC_soil'][1])

    for elem in ['N', 'P', 'K', 'Mg', 'Ca']:
        state_params[f'{elem}_soil'] += calculate_concentration_change(
            C_soil=state_params[f'{elem}_soil'], C_solution=state_params[f'{elem}_mix'],
            k_assim=np.random.uniform(k_assim[elem][0], k_assim[elem][1]), dt=dtime)

        state_params[f'{elem}_soil'] = np.clip(state_params[f'{elem}_soil'], low_high_dict[f'{elem}_soil'][0], low_high_dict[f'{elem}_soil'][1])

    if isProfile:
        profiler.disable()
        profiler.print_stats(sort='cumulative')
            
        print('END "_step_i_Soil" profile')

    return state_params, end_time, Volume + N1 + N2 + N3 - N_relief
