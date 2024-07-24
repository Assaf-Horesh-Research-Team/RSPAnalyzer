import pandas as pd
import numpy as np
import typing
from typing import Any,Tuple,List
import os
import glob  # Add this line to import the 'glob' module
from matplotlib import pyplot as plt
import scipy.optimize
import scipy.special as sf
import scipy.signal as sp
import scipy.linalg
from scipy.integrate import quad
from scipy.constants import electron_mass, Boltzmann, Planck
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize, differential_evolution, approx_fprime
import emcee
import time
from multiprocessing import Pool
import corner
import math
import tqdm
from emcee import EnsembleSampler
# from numpy.typing import NDArra

#constatns
C1 = 6.27e18 #cgs units - too big. maybe units err
# C1 = 6.27
C5 = 7.52e-24 #for gamma=3, cgs
C6 = 7.97e-41 #for gamma=3, cgs
F= 0.5 #emission filling factor of an approximate spherical volume with outter radius r
c = 3e10
Me = 9.1e-28
EL = Me * c**2 #cm*g/s
EPS_B = 0.1
EPS_E = 0.1
mJY = 1e26
MPC = 3.08e24
DAYS = 86400
Ghz = 1e9
eps_b=0.1
alpha = 1
m_electron = 9e-28 
q_electron = 4.8*1e-10
b=0.5
m_proton = 1.6726231*1e-24 #g
light_speed = 2.9*1e10


"""
This is RSPAnalyzer, a tool for initial parameter analysis of radio 
supernova spectra and light curves using the MCMC algorithm.
Each function has a corresponding version with _mn at the end of
 the name. These _mn functions are designed to fit only 4 parameters, 
 excluding m. The documentation is the same for both versions.
 """


def csv_file_to_dfs(path):
    """
    Reads a CSV file into a pandas DataFrame.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - df (pandas.DataFrame): DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(path)
    return df

def find_flux_time_peak_value(df, column_name:str)->tuple[Any,Any,Any]:
    """
    Finds the peak value and corresponding values in a specified column of a DataFrame.
    We use it in this library in order to find the peak of the measured flux from the fiven dataset,
    so we will able to calculate the initial guess to R0 and n0 parameters in the future.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data.
    - column_name (str): Name of the column to find the peak value.

    Returns:
    - tuple (Any, Any, Any): Tuple containing the peak value and corresponding values from the row.
      - First element: Value from the first column of the row with the peak value.
      - Second element: Value from the second column of the row with the peak value.
      - Third element: Value from the third column of the row with the peak value.
    """
    column_data = df[column_name]
    peak_col = column_data.idxmax()
    peak_row = df.iloc[peak_col]
    return peak_row[0],peak_row[1],peak_row[2]



def calc_init_guess_n_R(df: pd.DataFrame,column_name:str,alpha,d,eps_b)->tuple[Any,Any,Any]:
    """
    calculates initial guess for n0 and R0 according to Chavelier's 2006  (21), (22)
    finds the maximum flux from the measured data using find_flux_time_peak_value func 
    and converts it to density of csm and radius of the shockwave


    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data - flux in mJY and frequency in Ghz
    - column_name (str): Name of the column to find the peak value.
    - alpha (float): the ratio between epsilon_b (magnetic energy density) and 
    epsilon_e (relativistic electron energy density), here 1 by default (eps_b and eps_e both 0.1)

    Returns:
    - tuple (Any, Any, Any): Tuple containing:
    - First element: initial guess for the power of the shockwave's radius, R0.
    - Second element: initial guess for the power of the CSM's density number, n0.
    - Third element: time (in DAYS) of when the highest flux value was measured.
    """
    tp, fp,vp = find_flux_time_peak_value(df,column_name) #already in DAYS, mJY
    print("init func: ")
    print(tp/DAYS,fp/mJY,vp/Ghz)
    rp = 4*1e14*((alpha*(F/0.5))**(-1/19))*(fp**(9/19))*((d/MPC)**(18/19))*(((vp/Ghz)/5)**(-1)) #cm
    bp = 1.1*((alpha*(F/0.5))**(-4/19))*(fp**(-2/19))*((d/MPC)**(-4/19))*((vp/Ghz)/5) #gauss
    RHO0 = (bp/(9*np.pi*eps_b))*((tp/rp)**2)
    N0 = RHO0/m_proton
    # print("N0: ",N0)
    #make the density as the log of the result so the fitter will adjust correctly
    exponentN0 = int(math.log10(abs(N0)))
    if N0 < 1:
        exponentN0 -= 1

    return rp,exponentN0,tp

#mcmc private funcs

def _mult_fitted_function(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]],
                          t0: float,d:float,*theta)->float:
    
    """
    The model - the function according to which we sample the parameters, equation (1)
    from Chavalier's 1998 paper.

    Parameters
    ----------
    x : tuple of np.ndarray
        A tuple containing two numpy arrays:
        - t: An array of time values in DAYS
        - v: An array of velocity values in Ghz

    t0 : float
        The reference time parameter - here is the time of the peak in the flux from the dataset

    d : float
         distance from the supernova in MPC.

    *theta : tuple of floats
        The parameters we sample, which include:
        - R0: The initial radius exponent in cm.
        - m: The power-law index for the radius.
        - N0: The initial number density exponent.
        - k: The power-law index for the density.
        - gamma: The power-law index of the relatavistic particles density.

    Returns
    -------
    float
        The calculated value of the fitted function based on the given parameters.

    Notes
    -----
    This function is based on equation (1) from Chavalier's 1998 paper. It models the physical 
    properties of a supernova remnant, including the radius, velocity, density, magnetic field, 
    synchrotron emission, and flux density.
    """
    R0,m,N0, k,gamma= theta
    t,v = x
    r0 = 10**R0
    r=  r0*((t/t0)**m)
    vel = (m*r0/t0)*((t/t0)**(m-1))
    # print(vel)
    n = 10**N0
    rho = (n*m_proton)*((r/r0)**k)
    Usw= 9/8*rho*(vel**2)
    bfield = np.sqrt(8*np.pi*Usw*eps_b)

    n0 = (alpha* bfield**2 * (gamma-2) * EL**(gamma-2)) / (8*np.pi)
    pow = (2/(gamma+4))
    v1 = 2 * C1 * ((((4*F*r)/3)*C6)**pow)* (n0**pow) * (bfield**((gamma+2)/(gamma+4)))
    s_v1 = (C5/C6)*((bfield)**(-0.5))*((v1/(2*C1))**(5./2.))
    j= ((v/v1)**(5/2))*(1-np.exp(-(v/v1)**(-(gamma+4)/2)))
    iv = s_v1*j
    solid_angle = (np.pi*(r**2)/(d**2))
    return solid_angle*iv*mJY

def _mult_fitted_function_nm(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]],
                          t0: float,d:float,*theta)->float:
    R0,N0, k,gamma= theta
    m=1
    t,v = x
    r0 = 10**R0
    r=  r0*((t/t0)**m)
    vel = (m*r0/t0)*((t/t0)**(m-1))
    # print(vel)
    n = 10**N0
    rho = (n*m_proton)*((r/r0)**k)
    Usw= 9/8*rho*(vel**2)
    bfield = np.sqrt(8*np.pi*Usw*eps_b)

    n0 = (alpha* bfield**2 * (gamma-2) * EL**(gamma-2)) / (8*np.pi)
    pow = (2/(gamma+4))
    v1 = 2 * C1 * ((((4*F*r)/3)*C6)**pow)* (n0**pow) * (bfield**((gamma+2)/(gamma+4)))
    s_v1 = (C5/C6)*((bfield)**(-0.5))*((v1/(2*C1))**(5./2.))
    j= ((v/v1)**(5/2))*(1-np.exp(-(v/v1)**(-(gamma+4)/2)))
    iv = s_v1*j
    solid_angle = (np.pi*(r**2)/(d**2))
    return solid_angle*iv*mJY



def _mult_log_likelihood(theta: tuple[float, float, float, float, float], 
                         x: tuple[np.ndarray[Any, np.dtype[np.float64]], 
                                  np.ndarray[Any, np.dtype[np.float64]]], 
                         t0: float, 
                         d: float, 
                         flux: np.ndarray, 
                         eflux: np.ndarray) -> float: 
    """
    Calculate the log of the likelihood function for internal use in fitting procedures, 
    here its according to chi^2 model.

    This function is used to compute the log-likelihood by comparing the observed flux measurements
    with the model predictions generated by the `mult_fitted_function`.

    Parameters
    ----------
    theta : np.ndarray
        An array containing the free parameters to fit. These parameters are passed to the `mult_fitted_function`.

    x : tuple of np.ndarray
        A tuple containing two numpy arrays:
        - t: An array of time values.
        - v: An array of velocity values.

    t0 : float
        The reference time parameter - here is the time of the peak in the flux from the dataset

    d : float
         distance from the supernova in MPC.

    flux : np.ndarray
        The measurements of flux in mJY

    eflux : np.ndarray
        The uncertainty in the flux measurements in mJY

    Returns
    -------
    float
        The log of the likelihood function, which is used as a criterion for the goodness of fit.
    """
   
    model = _mult_fitted_function(x,t0,d,*theta)
    sigma = np.sqrt(eflux**2)
    chi2 = np.sum(((flux - model)/sigma)**2.)
    return -0.5*chi2

def _mult_log_likelihood_nm(theta, x, t0,d:float,flux, eflux): 
    # calculating the log of the likelihood function
    # ttheta is an array containing the free parameters to fit
    # x is the variables, in this case both time and frequency
    # flux is the measurments and eflux is its unceirtainty
    model = _mult_fitted_function_nm(x,t0,d,*theta)
    sigma = np.sqrt(eflux**2)
    chi2 = np.sum(((flux - model)/sigma)**2.)
    return -0.5*chi2

def _log_prior(theta):
    """
    Set the prior probabilities for the parameters in the model.

    This function defines the prior distributions for the parameters used in the model, based on 
    domain knowledge. It restricts the search space to specific ranges for each parameter. 
    If the parameters fall within the allowed ranges, the function returns 0.0, indicating 
    a uniform prior probability. If any parameter falls outside the allowed ranges, the function 
    returns negative infinity, indicating that the parameter combination is not allowed.
    If there is an actual known prior distribution to the parameters, you may want to modify this 
    function according to it.

    Parameters
    ----------
    theta : tuple[float, float, float, float, float]
        A tuple containing the parameters (R0, m, N0, k, gamma) to be evaluated.

    Returns
    -------
    float
        The log of the prior probability. Returns 0.0 if the parameters are within the allowed ranges, 
        and -inf if any parameter is outside the allowed ranges.

    Notes
    -----
    The allowed ranges for the parameters are:
    - R0: 14 to 17
    - m: 0.67 to 1
    - N0: 1 to 17
    - k: -5 to 0
    - gamma: 2 to 4

    These ranges are set based on prior knowledge and physical considerations 
    (more about it on README) and help in guiding the parameter search 
    within reasonable bounds.
    """

    R0,m,N0, k,gamma = theta
    if (14 < R0 < 17) and (0.67< m < 1) and (1 <N0< 17) and (-5 < k < 0) and (2. < gamma < 4.):
        # these are the allowed values
        return 0.0
    else:
        # if not, it return -inf so the next function will accomodate for it
        return -np.inf

def _log_prior_nm(theta):
    # setting the priors we want to use. Here I use some basic knowledge I have to set which are the allowed 
    # parameters to search in and make it easier. In real cases you might have to play with this
    R0,N0, k,gamma = theta
    if (14 < R0 < 17) and (1 <N0< 17) and (-5 < k < 0) and (2. < gamma < 4.):
        # these are the allowed values
        return 0.0
    else:
        # if not, it return -inf so the next function will accomodate for it
        return -np.inf
    
def _log_probability(theta: tuple[float, float, float, float, float], 
                     x: tuple[np.ndarray, np.ndarray], 
                     t0: float, 
                     d: float, 
                     y: np.ndarray, 
                     yerr: np.ndarray) -> float:
    """
    Calculate the logarithm of the probability for given parameters, data, and errors.

    This function combines the log prior and log likelihood to compute the log of the posterior 
    probability for the given parameters. If the prior probability is not finite (i.e., the parameters 
    are outside the allowed ranges), it returns negative infinity. Otherwise, it returns the sum of the 
    log prior and the log likelihood.

    Parameters
    ----------
    theta : tuple[float, float, float, float, float]
        A tuple containing the parameters (R0, m, N0, k, gamma) to be evaluated.
    x : tuple[np.ndarray, np.ndarray]
        A tuple containing the time and frequency arrays.
    t0 : float
        The initial time parameter.
    d : float
        Distance parameter.
    y : np.ndarray
        Observed data (flux measurements).
    yerr : np.ndarray
        Errors in the observed data (flux uncertainties).

    Returns
    -------
    float
        The log of the posterior probability. Returns -inf if the prior probability is not finite.
    """
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _mult_log_likelihood(theta, x,t0, d,y, yerr)

    
def _log_probability_nm(theta, x,t0, d,y, yerr):
    # calculating the log of the probability
    lp = _log_prior_nm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _mult_log_likelihood_nm(theta, x,t0, d,y, yerr)



def _max_likeli(initial: np.ndarray, 
                bounds: tuple[tuple[float, float], ...], 
                x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]], 
                t0: float, 
                d: float, 
                y: np.ndarray, 
                yerr: np.ndarray) -> scipy.optimize.OptimizeResult:
    """
    Maximize the likelihood function to find the best-fit parameters.

    This function uses the Nelder-Mead algorithm to maximize the likelihood function, starting from an 
    initial guess and within given parameter bounds. It returns the solution found by the optimizer.
    The maximum likelihood will be the Markov chain's starting point.
    NOTICE: here the maximization fails from some reason and get returns the initial guess.

    Parameters
    ----------
    initial : tuple[float, float, float, float, float]
        Initial guess for the parameters.
    bounds : tuple[tuple[float, float], ...]
        Bounds for the parameters during optimization.
    x : tuple[np.ndarray, np.ndarray]
        A tuple containing the time and frequency arrays.
    t0 : float
        The initial time parameter.
    d : float
        Distance parameter.
    y : np.ndarray
        Observed data (flux measurements).
    yerr : np.ndarray
        Errors in the observed data (flux uncertainties).

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of the optimization process. Contains information such as the optimized parameters 
        and the success status of the optimization.
    """
    # maximize the likelihood function
    np.random.seed(42)
    def nll(*args):
        return -_mult_log_likelihood(*args)
    soln = minimize(nll, initial, args=(x, t0,d,y, yerr),bounds=bounds)
    return soln

def _max_likeli_nm(initial,bounds,x,t0,d,y,yerr):
    # maximize the likelihood function
        np.random.seed(42)
        def nll(*args):
            return -_mult_log_likelihood_nm(*args)
        soln = minimize(nll, initial, args=(x, t0,d,y, yerr),bounds=bounds)
        # # result_de = differential_evolution(nll, bounds, args=(x, t0, d, y, yerr))
        # # initial_guess = result_de.x
        # soln=scipy.optimize.basinhopping(nll, initial, niter=100,minimizer_kwargs={'args':(x, t0,d,y, yerr),'method':'L-BFGS-B'},disp=True)
        return soln

def calc_likeli_param(x:tuple[np.ndarray[Any, np.dtype[np.float64]],
                              np.ndarray[Any, np.dtype[np.float64]]],
                      y: np.ndarray, 
                      yerr: np.ndarray, 
                      d: float, 
                      k: float, 
                      m: float, 
                      gamma: float) -> list[float]:
    """
    Calculate the maximum likelihood parameters for the given data.

    This function calculates the maximum likelihood estimates of the parameters for a given dataset. 
    It takes initial guesses for the parameters, sets their bounds, and uses the maximum likelihood 
    estimation to find the best-fit parameters.

    Parameters
    ----------
    x : tuple[list[float], list[float]]
        A tuple containing two lists: time vector and frequency vector.
    y : list[float]
        Observed flux measurements.
    yerr : list[float]
        Uncertainties in the observed flux measurements.
    d : float
        Distance parameter.
    k : float
        Initial guess for the k parameter.
    m : float
        Initial guess for the m parameter.
    gamma : float
        Initial guess for the gamma parameter.

    Returns
    -------
    list[float]
        The maximum likelihood estimates of the parameters: [R0, m, N0, k, gamma].
    """
    t_vec, v_vec = x
    df = pd.DataFrame({'delta_t': t_vec, 'flux': y, 'freq': v_vec})
    R0,N0,tp =calc_init_guess_n_R(df,'flux', alpha,d,eps_b) #take the initial guess from 1.43GHz
    # R0,m,RHO0, k,gamma
    init = np.array([np.log10(R0),m,N0,k,gamma]) # initial parameters
    # init = np.array([np.log10(R0),m,9,k,gamma]) # initial parameters

    print("init_guess", init)
    bnds = ((14,17),(0.67,1),(1,17),(-5,0),(2,4)) # parameters bounds
    soln = _max_likeli(initial=init,bounds=bnds,x=x,t0=tp*DAYS,d=d,y=y,yerr=yerr) # solving
    print("sol", soln.x)
    fp_ml, tp_ml, a_ml, b_ml, p_ml = soln.x # Extracting best fit parameteres
    # printing
    print("Maximum likelihood estimates:")
    print("R0 = {0:.3f}".format(fp_ml))
    print("m = {0:.3f}".format(tp_ml))
    print("RHO0 = {0:.3f}".format(a_ml))
    print("k = {0:.3f}".format(b_ml))
    print("gamma = {0:.3f}".format(p_ml))
    return soln.x

def calc_likeli_param_nm(x:tuple[list[float],list[float]],y:list[float],yerr:list[float],d:float,k:float, m:float, gamma:float)-> list[float]:
    t_vec, v_vec = x
    df = pd.DataFrame({'delta_t': t_vec, 'flux': y, 'freq': v_vec})
    R0,N0,tp =calc_init_guess_n_R(df,'flux', alpha,d,eps_b) #take the initial guess from 1.43GHz
    # R0,m,RHO0, k,gamma
    init = np.array([np.log10(R0),N0,k,gamma]) # initial parameters
    # init = np.array([np.log10(R0),m,9,k,gamma]) # initial parameters

    print("init_guess", init)
    bnds = ((14,17),(1,17),(-5,0),(2,4)) # parameters bounds
    soln = _max_likeli_nm(initial=init,bounds=bnds,x=x,t0=tp*DAYS,d=d,y=y,yerr=yerr) # solving
    print("sol", soln.x)
    fp_ml, a_ml, b_ml, p_ml = soln.x # Extracting best fit parameteres
    # printing
    print("Maximum likelihood estimates:")
    print("R0 = {0:.3f}".format(fp_ml))
    print("RHO0 = {0:.3f}".format(a_ml))
    print("k = {0:.3f}".format(b_ml))
    print("gamma = {0:.3f}".format(p_ml))
    return soln.x


def run_mcmc_get_sampler(x:tuple[np.ndarray[Any, np.dtype[np.float64]],
                              np.ndarray[Any, np.dtype[np.float64]]],
                         y: np.ndarray, 
                         yerr: np.ndarray, 
                         tp:float,
                         d:float,
                         nwalkers:int,
                         ndim:int,
                         nsteps:int,
                         solnx:list[float])->EnsembleSampler:
    """
    Run the MCMC process to sample from the posterior distribution.

    This function initializes the positions of the MCMC walkers and runs the MCMC process in parallel 
    using the `emcee` library to sample from the posterior distribution of the model parameters.

    Parameters
    ----------
    x : tuple[list[float], list[float]]
        A tuple containing two lists: time vector and frequency vector.
    y : list[float]
        Observed flux measurements.
    yerr : list[float]
        Uncertainties in the observed flux measurements.
    tp : float
        Characteristic time parameter- time of the peak.
    d : float
        Distance parameter.
    nwalkers : int
        Number of walkers in the MCMC.
    ndim : int
        Number of dimensions (parameters) to sample.
    nsteps : int
        Number of steps for each walker - length of each Markov chain on the ensamble.
        Each Markov chain represents different parameter.
    solnx : list[float]
        Initial guess for the parameters - the starting .

    Returns
    -------
    emcee.EnsembleSampler
        The sampler object containing the MCMC samples.

    Notes
    -----
    - The function initializes the positions of the MCMC walkers around the initial guess `solnx`.
    - It runs the MCMC sampler in parallel using a multiprocessing pool.
    - The progress of the MCMC sampling is displayed and the total time taken is printed.
    """
    pos = solnx + 1e-4 * np.random.randn(nwalkers,ndim) # initial position of the chain
    # Running in parallel
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability, args=(x, tp*DAYS, d,y,yerr),pool=pool) # setting posterior sampler
        start = time.time()
        sampler.run_mcmc(initial_state=pos, nsteps=nsteps, progress=True) # run mcmc sampler
        end = time.time()
        multi_time = end - start
        print("Serial took {0:.1f} seconds".format(multi_time))
    return sampler

def run_mcmc_get_sampler_nm(x:tuple[list[float],list[float]],y:list[float],yerr:list[float],tp:float,d:float,nwalkers:int,ndim:int,nsteps:int,solnx:list[float])->EnsembleSampler:
    # Running the MCMC process
    pos = solnx + 1e-4 * np.random.randn(nwalkers,ndim) # initial position of the chain
    # Running in parallel
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability_nm, args=(x, tp*DAYS, d,y,yerr),pool=pool) # setting posterior sampler
        start = time.time()
        sampler.run_mcmc(initial_state=pos, nsteps=nsteps, progress=True) # run mcmc sampler
        end = time.time()
        multi_time = end - start
        print("Serial took {0:.1f} seconds".format(multi_time))
    return sampler

def extract_mcmc_params(sampler: EnsembleSampler)->list[float]:
    """
    Extract the median values of the MCMC samples for each parameter.
    This function retrieves the posterior samples from the MCMC sampler, 
    computes the median value of each parameter, and returns these medians as a list.
    - The function discards the first 100 samples as burn-in, and thins the chain by a factor of 15.
    - The median values of the parameters are computed from the thinned, flattened samples.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object containing the posterior samples.

    Returns
    -------
    list[float]
        A list containing the median values of the parameters:
        [R0_mcmc, m_mcmc, N0_mcmc, k_mcmc, gamma_mcmc].

    """
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    R0_mcmc = np.median(flat_samples[:,0])
    m_mcmc = np.median(flat_samples[:,1])
    N0_mcmc = np.median(flat_samples[:,2])
    k_mcmc = np.median(flat_samples[:,3])
    gamma_mcmc = np.median(flat_samples[:,4])
    print(R0_mcmc,m_mcmc,N0_mcmc,k_mcmc,gamma_mcmc)
    return [R0_mcmc,m_mcmc,N0_mcmc,k_mcmc,gamma_mcmc]


def extract_mcmc_params_nm(sampler: EnsembleSampler)->list[float]:
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    R0_mcmc = np.median(flat_samples[:,0])
    N0_mcmc = np.median(flat_samples[:,1])
    k_mcmc = np.median(flat_samples[:,2])
    gamma_mcmc = np.median(flat_samples[:,3])
    print(R0_mcmc,N0_mcmc,k_mcmc,gamma_mcmc)
    return [R0_mcmc,N0_mcmc,k_mcmc,gamma_mcmc]

#error calculations and visualization

def calc_params_err(sampler: EnsembleSampler, mcmc_param: list):
    """
    Calculate the parameter uncertainties from MCMC samples.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object containing the posterior samples.

    mcmc_param : list
        A list of MCMC parameters. Used here to determine the number of parameters and their order.

    Returns
    -------
    list[tuple[float, float, float]]
        A list of tuples, each containing:
        - The lower uncertainty (16th percentile - median).
        - The median value (50th percentile).
        - The upper uncertainty (84th percentile - median) for each parameter.
    """
    burnin = 1000
    samples = sampler.get_chain(discard=burnin, flat=True) 
    err_arr = [] 
    theta_mcmc = np.percentile(samples, [16, 50, 84], axis=0)
    theta_median = theta_mcmc[1]
    theta_lower = theta_mcmc[1] - theta_mcmc[0]
    theta_upper = theta_mcmc[2] - theta_mcmc[1]
    ndim= len(mcmc_param)
    n = ["R0", "m: radius power law", "N0", "k: num density power law", "gamma: energy power law"]
    for i, j in zip(range(ndim),n):
        err_arr.append((theta_lower[i],theta_median[i],theta_upper[i]))
    return err_arr

def keep_significant_digits(num1, num2):
    """
    Format a number to keep significant digits.

    This function formats `num1` to a number of significant digits that is appropriate given
    the precision of `num2`. It attempts to match the number of decimal places in `num2`.

    Parameters
    ----------
    num1 : float
        The number to be formatted - the mean value of the parameter.
        
    num2 : float
        The reference number whose precision determines the formatting of `num1`- 
        the parameter's uncertainties
        
    Returns
    -------
    float
        The value of the parameters with errors.
    """
    num2_str = str(num2)
    decimal_found = False
    digits_to_keep = 0
    for char in num2_str:
        if char == '.':
            decimal_found = True
            continue
        if decimal_found and char != '0':
            break
        if decimal_found:
            digits_to_keep += 1
        format_str = "{:." + str(digits_to_keep + 1) + "f}"
    formatted_num1 = format_str.format(num1)
    ret = float(formatted_num1)
    
    return ret


#plots_funcs

def plot_light_curve(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]], 
                     y:np.ndarray, yerr:np.ndarray,xlim:tuple[float, float],
                     ylim:tuple[float, float],tp:float,d:float,mcmc_params,
                     flat_samples,legend:bool):
    """
    Plot the light curve of the observed data and model predictions. log-log scale.

    This function plots the flux as a function of time for different frequencies, along with
    the fitted model and its MCMC samples. The plot includes error bars for the flux measurements.

    Parameters
    ----------
    x : tuple[np.ndarray[Any, np.dtype[np.float64]]
        A tuple containing:
        - `t_vec`: Array of time values in DAYS
        - `v_vec`: Array of frequency values in Ghz
        
    y : np.ndarray
        List of observed flux values in mJY

    yerr : np.ndarray
        List of uncertainties in the observed flux values in mJY

    xlim : tuple[float, float]
        X-axis limits for the plot.

    ylim : tuple[float, float]
        Y-axis limits for the plot.

    tp : float
        Time of peak flux (or some relevant time point) used for fitting.

    d : float
        Distance to the source.

    mcmc_params : list[float]
        Best-fit parameters obtained from MCMC sampling.

    flat_samples : np.ndarray
        Array of flattened MCMC samples used to plot the uncertainty bands.

    legend : bool
        Whether to display the legend on the plot.
    """
    t_vec, v_vec = x
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1, 1, 1)
    uniq_freq = np.unique(v_vec/Ghz)
    for tmpfreq in uniq_freq:
        idxs = np.where(v_vec/Ghz==tmpfreq)[0]
        temp_t = (t_vec/DAYS)[idxs]
        temp_flux = y[idxs]
        temp_eflux = yerr[idxs]
        ax.errorbar(temp_t,temp_flux,temp_eflux,
                    linestyle='none',marker='o',markersize='8',mec='k',alpha=0.5,
                    label=str(float(tmpfreq)) + ' GHz',zorder=1000)
        temp_t_fit = np.geomspace(1.,2000.,1000)*DAYS
        temp_nu_fit = tmpfreq*Ghz*np.ones(len(temp_t_fit))
        temp_x_fit = (temp_t_fit,temp_nu_fit)
        temp_theta_fit = mcmc_params
        temp_y_fit = _mult_fitted_function(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_t_fit/DAYS,temp_y_fit,color=plt.gca().lines[-1].get_color())

        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = _mult_fitted_function(temp_x_fit,tp*DAYS,d,*sample)
            ax.plot(temp_t_fit/DAYS, tempf_fit, alpha=0.02,color=plt.gca().lines[-1].get_color(),zorder=10)
        
        
    ax.set_xscale('log')
    ax.set_xlabel('Days since explosion',fontsize=20)
    ax.set_ylabel(r'${F_{\rm \nu}} \, \rm \left[ mJy \right]$',fontsize=20)
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        ax.legend(loc='lower right',fontsize=12)
    if xlim and ylim: 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def plot_light_curve_nm(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]], 
                     y:np.ndarray, yerr:np.ndarray,xlim:tuple[float, float],
                     ylim:tuple[float, float],tp:float,d:float,mcmc_params,
                     flat_samples,legend:bool):
    t_vec, v_vec = x
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1, 1, 1)
    uniq_freq = np.unique(v_vec/Ghz)
    for tmpfreq in uniq_freq:
        idxs = np.where(v_vec/Ghz==tmpfreq)[0]
        temp_t = (t_vec/DAYS)[idxs]
        temp_flux = y[idxs]
        temp_eflux = yerr[idxs]
        ax.errorbar(temp_t,temp_flux,temp_eflux,
                    linestyle='none',marker='o',markersize='8',mec='k',alpha=0.5,
                    label=str(float(tmpfreq)) + ' GHz',zorder=1000)
        temp_t_fit = np.geomspace(1.,2000.,1000)*DAYS
        temp_nu_fit = tmpfreq*Ghz*np.ones(len(temp_t_fit))
        temp_x_fit = (temp_t_fit,temp_nu_fit)
        temp_theta_fit = mcmc_params
        temp_y_fit = _mult_fitted_function_nm(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_t_fit/DAYS,temp_y_fit,color=plt.gca().lines[-1].get_color())

        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = _mult_fitted_function_nm(temp_x_fit,tp*DAYS,d,*sample)
            ax.plot(temp_t_fit/DAYS, tempf_fit, alpha=0.02,color=plt.gca().lines[-1].get_color(),zorder=10)
        
        
    ax.set_xscale('log')
    ax.set_xlabel('Days since explosion',fontsize=20)
    ax.set_ylabel(r'${F_{\rm \nu}} \, \rm \left[ mJy \right]$',fontsize=20)
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        ax.legend(loc='lower right',fontsize=12)
    if xlim and ylim: 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def plot_spectra(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]], 
                y:list[float],yerr:list[float],xlim:tuple[float, float],
                ylim:tuple[float, float],tp:float,d:float,mcmc_params,flat_samples,legend:bool):
    """
    Plot the spectra of the sampled radio supernova on log-log scale.

    This function plots the flux as a function of frequency for different times, along with
    the fitted model and its MCMC samples. The plot includes error bars for the flux measurements.
     Parameters
    ----------
    x : tuple[np.ndarray[Any, np.dtype[np.float64]]
        A tuple containing:
        - `t_vec`: Array of time values in DAYS
        - `v_vec`: Array of frequency values in Ghz
        
    y : np.ndarray
        List of observed flux values in mJY

    yerr : np.ndarray
        List of uncertainties in the observed flux values in mJY

    xlim : tuple[float, float]
        X-axis limits for the plot.

    ylim : tuple[float, float]
        Y-axis limits for the plot.

    tp : float
        Time of peak flux (or some relevant time point) used for fitting.

    d : float
        Distance to the source.

    mcmc_params : list[float]
        Best-fit parameters obtained from MCMC sampling.

    flat_samples : np.ndarray
        Array of flattened MCMC samples used to plot the uncertainty bands.

    legend : bool
        Whether to display the legend on the plot.
    """
    t_vec, v_vec = x
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(1, 1, 1)
    uniq_time = np.unique(t_vec/DAYS)
    for tmpt in uniq_time:
        idxs = np.where(t_vec/DAYS==tmpt)[0]
        temp_nu = (v_vec/Ghz)[idxs]
        temp_flux = y[idxs]
        temp_eflux = yerr[idxs]
        ax.errorbar(temp_nu,temp_flux,temp_eflux,
                    linestyle='none',marker='o',markersize='8',mec='k',alpha=0.8,
                    label=str(float(tmpt)) + ' DAYS',zorder=1000)
     
        temp_nu_fit = np.geomspace(1.,35,200)*Ghz
        temp_t_fit = tmpt*DAYS*np.ones(len(temp_nu_fit))
        temp_x_fit = (temp_t_fit,temp_nu_fit)
        temp_theta_fit = mcmc_params
        temp_y_fit = _mult_fitted_function(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_nu_fit/Ghz,temp_y_fit,color=plt.gca().lines[-1].get_color())
        
        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = _mult_fitted_function(temp_x_fit,tp*DAYS,d,*sample)
            ax.plot(temp_nu_fit/Ghz, tempf_fit, alpha=0.02,color=plt.gca().lines[-1].get_color(),zorder=10)
     
        
    ax.set_xscale('log')
    ax.set_xlabel('frequency [Ghz]',fontsize=20)
    ax.set_ylabel(r'${F_{\rm \nu}} \, \rm \left[ mJy \right]$',fontsize=20)
    ax.set_yscale('log')
    if legend:
        ax.legend(loc='lower right',fontsize=12)
    if xlim and ylim: 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def plot_spectra_nm(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]], 
                y:list[float],yerr:list[float],xlim:tuple[float, float],
                ylim:tuple[float, float],tp:float,d:float,mcmc_params,flat_samples,legend:bool):
    t_vec, v_vec = x
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(1, 1, 1)
    # err = np.sqrt(np.sum((0.10*y)**2+yerr**2))
    uniq_time = np.unique(t_vec/DAYS)
    for tmpt in uniq_time:
        idxs = np.where(t_vec/DAYS==tmpt)[0]
        # temp_t = (t_vec/DAYS)[idxs]
        temp_nu = (v_vec/Ghz)[idxs]
        temp_flux = y[idxs]
        temp_eflux = yerr[idxs]
        ax.errorbar(temp_nu,temp_flux,temp_eflux,
                    linestyle='none',marker='o',markersize='8',mec='k',alpha=0.8,
                    label=str(float(tmpt)) + ' DAYS',zorder=1000)
        # temp_t_fit = np.geomspace(1.,2000.,1000)*DAYS
        # temp_nu_fit = tmpt*Ghz*np.ones(len(temp_t_fit))
        temp_nu_fit = np.geomspace(1.,35,200)*Ghz
        temp_t_fit = tmpt*DAYS*np.ones(len(temp_nu_fit))
        temp_x_fit = (temp_t_fit,temp_nu_fit)
        temp_theta_fit = mcmc_params
        temp_y_fit = _mult_fitted_function_nm(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_nu_fit/Ghz,temp_y_fit,color=plt.gca().lines[-1].get_color())
        
        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = _mult_fitted_function_nm(temp_x_fit,tp*DAYS,d,*sample)
            ax.plot(temp_nu_fit/Ghz, tempf_fit, alpha=0.02,color=plt.gca().lines[-1].get_color(),zorder=10)
     
        
    ax.set_xscale('log')
    ax.set_xlabel('frequency [Ghz]',fontsize=20)
    ax.set_ylabel(r'${F_{\rm \nu}} \, \rm \left[ mJy \right]$',fontsize=20)
    ax.set_yscale('log')
    if legend:
        ax.legend(loc='lower right',fontsize=12)
    if xlim and ylim: 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

# func for swapping from eq4 to eq1 
def eq4_to_eq1_powerlaws_conversion(a,b,gamma):
    """
    Convert power-law parameters from Equation 4 to Equation 1 of the model. (Chavalier's 1998 paper)
    This function computes the parameters `m` and `k` used in Equation 1 
    based on the input parameters `a`, `b`, and `gamma` from Equation 4.
    
    Parameters
    ----------
    a : float
        Parameter `a` from Equation 4 - power law of the optically thick regime

    b : float
        Parameter `b` from Equation 4 - power law of the optically thick regime

    gamma : float
        gamma: The power-law index of the relatavistic particles density.

    Returns
    -------
    tuple[float, float] 
        A tuple containing the computed parameters `m` and `k` for Equation 1.
        - m: The power-law index for the radius.
        - k: The power-law index for the density.
    """
    m= (-4*b+4*a*gamma+20*a+2*gamma+10)/(62+10*gamma)
    k = 6-(4*a-2)/m
    return m,k

# mass loss calculation - correct only when k=-2
def calc_mass_loss(mcmc_param: list, tp: float, t_vec: list):
    """
    Calculate the mass loss rates based on the model parameters and wind velocities (in km/h)
    This function computes the mass loss rates for different wind velocities using the given model parameters. It prints the mass loss rates for three specified wind velocities.
    The calculation is performed under the assumption that `k = -2`.
    Parameters
    ----------
    mcmc_param : list[float]
        List containing model parameters:
        - `R0` : Initial radius (log scale).
        - `m` : Radius power-law index.
        - `N0` : Number density (log scale).
        - `k` : Number density power-law index.
        - `gamma` : Energy power-law index.

    tp : float
        Time of peak flux (or some relevant time point) used for calculation.

    t_vec : list[float]
        Array of time values (not used in the current implementation).
    """
    R0, m, N0, k, gamma = mcmc_param
    v_wind = [10,1e2,1e3] #(km/s)
    RHO0 = (10**N0)*m_proton
    mass_losses = []
    for w in v_wind:
        massLoss_divided_vw =((3/2)*1e-21)* ((4*np.pi*RHO0)*((10**R0)**2)) #massLoss/year per km/s
        mass_losses.append(massLoss_divided_vw*w)
    for i,j in zip(v_wind,mass_losses):
        print(f"V_wind: {i} (km per sec), mass_loss: {j} (solar mass per year)")
    

#funcs for playing with epochs

def show_epocs(df:pd.DataFrame):
    """
    Display the unique time and frequency epochs from the given DataFrame.
    This function prints out the unique time epochs and frequency epochs 
    found in the DataFrame. It assumes that the DataFrame contains columns 
    named "delta_t" and "freq" representing time and frequency values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing columns "delta_t" and "freq".
    """
    t_epocs = np.unique(np.asarray(df["delta_t"].values))
    v_epocs = np.unique(np.asarray(df["delta_t"].values))
    print("time epocs (DAYS): {}".format(t_epocs))
    print("freq epocs (Ghz): {}".format(v_epocs))

def remove_epoc(df:pd.DataFrame, epoch: float,col_name: str)->pd.DataFrame:
    """
    Remove rows from the DataFrame where the specified column matches the given epoch.
    This function filters out rows from the DataFrame where the value in the 
    specified column equals the given epoch. This is useful for excluding 
    specific epochs from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to remove rows.

    epoch : float
        The value in the specified column that indicates which rows to remove.

    col_name : str
        The name of the column to check for the epoch value.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the rows removed where the column matches the epoch.
    """
   
    new_df = df[df[col_name] != epoch]
    return new_df

def single_epoch(df:pd.DataFrame, epoch: float,col_name: str)->pd.DataFrame:
    """
    Filter the DataFrame to include only rows where the specified column matches the given epoch.
    This function returns a DataFrame consisting of rows where the value in 
    the specified column equals the given epoch. This is useful for selecting 
    data corresponding to a specific epoch.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.

    epoch : float
        The value in the specified column that indicates which rows to include.

    col_name : str
        The name of the column to check for the epoch value.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the rows where the column matches the epoch.

    """
    new_df = df[df[col_name] == epoch]
    return new_df

# func for running the program all together. 
def run_prog(csv_path: str,
             dist_to_sn: float,
             init_k: float, 
             init_m: float, 
             init_gamma:float, 
             nwalkers: int,
             nsteps: int,
             fit_m: bool,
             spectra: bool, 
             light_curve: bool, 
             slegend,
             lclegend,
             xlim=None,
             ylim=None):
    """
    Run the full analysis pipeline for fitting and visualizing supernova data from a CSV file.

    This function performs the following steps:
    1. Loads the data from a CSV file.
    2. Computes errors and prepares the data for analysis.
    3. Calculates initial parameter guesses.
    4. Runs an MCMC sampler to fit the model.
    5. Extracts and processes the MCMC results.
    6. Optionally generates and displays light curve and spectra plots.
    7. Generates a corner plot of the parameter distributions.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing supernova data.

    dist_to_sn : float
        Distance to the supernova. Determines the distance unit (e.g., in Mpc).

    init_k : float
        Initial guess for the parameter `k` in the model.

    init_m : float
        Initial guess for the parameter `m` in the model.

    init_gamma : float
        Initial guess for the parameter `gamma` in the model.

    nwalkers : int
        Number of walkers for the MCMC sampler.

    nsteps : int
        Number of steps to run in the MCMC sampler.

    fit_m : bool
        wheather m sampled or not

    spectra : bool
        If True, plots the spectra.

    light_curve : bool
        If True, plots the light curve.

    slegend : bool
        If True, includes a legend in the spectra plot.

    lclegend : bool
        If True, includes a legend in the light curve plot.

    xlim : tuple[float, float], optional
        Limits for the x-axis of the plots. If None, default limits are used.

    ylim : tuple[float, float], optional
        Limits for the y-axis of the plots. If None, default limits are used.

    Returns
    -------
    Tuple
        A tuple containing:
        - `sampler` (emcee.EnsembleSampler): The MCMC sampler instance.
        - `mcmc_params` (list[float]): The best-fit parameters obtained from the MCMC sampling.
        - `flat_samples` (np.ndarray): The flattened array of samples from the MCMC chains.

    Notes
    -----
    - The function assumes that the CSV file contains columns named 'freq', 'delta_t', 'flux', and 'flux_err'.
    - The function uses a fixed seed for reproducibility of the MCMC results.
    - The MCMC sampling is performed in parallel using the `emcee` package.
    - If either `spectra` or `light_curve` is True, the function will generate corresponding plots.
    """
    # dist_to_sn  - decide wheather in Mpc or not
    df= csv_file_to_dfs(csv_path)
    df["err"]=[np.sqrt((0.1*f)**2+e**2) for f,e in zip(df["flux"].values,df["flux_err"].values)]

    v_vec = df['freq'].values*Ghz
    t_vec =df['delta_t'].values*DAYS

    y = df['flux'].values 
    # yerr =  df['flux_err'].values 
    yerr = df['flux_err'].values 
    # yerr =  df['err'].values*
    x = (t_vec, v_vec)
    d=dist_to_sn
    tp,fp,vp = find_flux_time_peak_value(df,'flux')
    if fit_m:
        solnx = calc_likeli_param(x=x,y=y,yerr=yerr,d=d,k=init_k,m=init_m,gamma=init_gamma)
        sampler = run_mcmc_get_sampler(x=x,y=y,yerr=yerr,tp=tp,d=d,nwalkers=nwalkers,ndim=5,nsteps=nsteps,solnx=solnx)
        flat_samples = sampler.get_chain(discard=1000, thin=20, flat=True)
        mcmc_params = extract_mcmc_params(sampler)
        if light_curve:
            plot_light_curve(x=x,y=y,yerr=yerr,xlim=xlim,ylim=ylim,tp=tp,d=d,mcmc_params=mcmc_params,flat_samples=flat_samples,legend=lclegend)
        if spectra:
            plot_spectra(x=x,y=y,yerr=yerr,xlim=xlim,ylim=ylim,tp=tp,d=d,mcmc_params=mcmc_params, flat_samples=flat_samples,legend=slegend)
        fig= corner.corner(flat_samples[1:],labels=[r'$RO_p$',r'$m$',r'$RHO0$',r'$k$',r'$gamma$'],bins=100,   label_kwargs={"fontsize": 20}, 
                        hist_kwargs={"histtype": "step", "linewidth": 1.5},
                            show_titles=True,plot_density=False,plot_datapoints=False)
        return sampler, mcmc_params, flat_samples
    else:
        solnx = calc_likeli_param_nm(x=x,y=y,yerr=yerr,d=d,k=init_k,m=init_m,gamma=init_gamma)
        sampler = run_mcmc_get_sampler_nm(x=x,y=y,yerr=yerr,tp=tp,d=d,nwalkers=nwalkers,ndim=4,nsteps=nsteps,solnx=solnx)
        flat_samples = sampler.get_chain(discard=1000, thin=20, flat=True)
        mcmc_params = extract_mcmc_params_nm(sampler)
        if light_curve:
            plot_light_curve_nm(x=x,y=y,yerr=yerr,xlim=xlim,ylim=ylim,tp=tp,d=d,mcmc_params=mcmc_params,flat_samples=flat_samples,legend=lclegend)
        if spectra:
            plot_spectra_nm(x=x,y=y,yerr=yerr,xlim=xlim,ylim=ylim,tp=tp,d=d,mcmc_params=mcmc_params, flat_samples=flat_samples,legend=slegend)
        fig= corner.corner(flat_samples[1:],labels=[r'$RO_p$',r'$n0$',r'$k$',r'$gamma$'],bins=100,   label_kwargs={"fontsize": 20}, 
                        hist_kwargs={"histtype": "step", "linewidth": 1.5},
                            show_titles=True,plot_density=False,plot_datapoints=False)
        return sampler, mcmc_params, flat_samples
