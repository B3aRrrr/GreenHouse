import numpy as np
from scipy import integrate

def calc_T_soil(
    dt:float,T_soil_0:float,
    R:float=0.35,h:float=0.35,
    T_air:float=299,
    _lambda:float=np.random.uniform(0.8,1.16)) -> np.ndarray(shape=(1,),dtype=np.float64):
    """
        Данный метод используется для расчета 
        точного значения средней температуры в 
        цилиндрическом горшке с радиусом R и высотой h. 
        Формула учитывает разницу в начальной температуре 
        Tₐ и конечной температуре Tᵢ, 
        а также время t, прошедшее
        с момента начала нагрева или охлаждения.

        Args:
            dt (float): Time. Measured in sec.
            T_air (float): Soi temperature (old). Measured in K;
            R (float): Radius of the cylindrical pot. Measured in meters. Default to 0.35; 
            h (float): Height of the cylindrical pot. Measured in meters. Default to 0.35; 
            T_air (float): Air temperature. Measured in K. Defaults to 26+273;
            _lambda(float): Coefficient of thermal conductivity. Measured in W/(m*K). Defaults to np.random.uniform(0.8,1.16);
        Result:
            np.ndarray(shape=(1,),dtype=np.float64) Calculated temperature of soil in the pot and an estimate of the error.
    """
         
    f = lambda r, theta: T_air + (T_soil_0 - T_air) * np.sin(theta) * r/np.exp(_lambda**2 * dt)
    mu,sigma = 1/(2*np.pi*R*h)*np. asarray(integrate.dblquad(f, 0, R, 0, 2*np.pi))
    return np.random.normal(mu, sigma, 1)

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
