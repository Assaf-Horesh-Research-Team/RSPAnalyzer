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
from numpy.typing import NDArra

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




def csv_file_to_dfs(path):
    df = pd.read_csv(path)
    return df

def find_flux_time_peak_value(df, column_name:str)->tuple[Any,Any,Any]:
    column_data = df[column_name]
    peak_col = column_data.idxmax()
    peak_row = df.iloc[peak_col]
    return peak_row[0],peak_row[1],peak_row[2]

def flux_all_at_once(t,v,d,t0,R0,m,N0, k,gamma)->float:
    r0 = 10**R0
    r=  r0*((t/t0)**m)
    vel = (m*r0/t0)*((t/t0)**(m-1))
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


def calc_init_guess_n_R(df: pd.DataFrame,column_name:str,alpha,d,eps_b)->tuple[any,any,any]:
  tp, fp,vp = find_flux_time_peak_value(df,column_name) #already in DAYS, mJY
  print(tp/DAYS,fp/mJY,vp/Ghz)
  rp = 4*1e14*((alpha*(F/0.5))**(-1/19))*(fp**(9/19))*((d/MPC)**(18/19))*(((vp/Ghz)/5)**(-1)) #cm
  bp = 1.1*((alpha*(F/0.5))**(-4/19))*(fp**(-2/19))*((d/MPC)**(-4/19))*((vp/Ghz)/5) #gauss
  RHO0 = (bp/(9*np.pi*eps_b))*((tp/rp)**2)
  # print("RHO0: ",RHO0)
  N0 = RHO0/m_proton
  # print("N0: ",N0)
  #make the density as the log of the result so the fitter will adjust correctly
  exponentN0 = int(math.log10(abs(N0)))
  if N0 < 1:
      exponentN0 -= 1

  return rp,exponentN0,tp

#mcmc private funcs

def mult_fitted_function(x:tuple[np.ndarray[Any, np.dtype[np.float64]],np.ndarray[Any, np.dtype[np.float64]]],t0: float,d:float,*theta)->float:
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

def mult_log_likelihood(theta, x, t0,d:float,flux, eflux): 
    # calculating the log of the likelihood function
    # ttheta is an array containing the free parameters to fit
    # x is the variables, in this case both time and frequency
    # flux is the measurments and eflux is its unceirtainty
    model = mult_fitted_function(x,t0,d,*theta)
    sigma = np.sqrt(eflux**2)
    chi2 = np.sum(((flux - model)/sigma)**2.)
    return -0.5*chi2

def log_prior(theta):
    # setting the priors we want to use. Here I use some basic knowledge I have to set which are the allowed 
    # parameters to search in and make it easier. In real cases you might have to play with this
    R0,m,N0, k,gamma = theta
    if (14 < R0 < 17) and (0.67< m < 1) and (1 <N0< 17) and (-5 < k < 0) and (2. < gamma < 4.):
        # these are the allowed values
        return 0.0
    else:
        # if not, it return -inf so the next function will accomodate for it
        return -np.inf
def log_probability(theta, x,t0, d,y, yerr):
    # calculating the log of the probability
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + mult_log_likelihood(theta, x,t0, d,y, yerr)

def max_likeli(initial,bounds,x,t0,d,y,yerr):
    # maximize the likelihood function
        np.random.seed(42)
        def nll(*args):
            return -mult_log_likelihood(*args)
        soln = minimize(nll, initial, args=(x, t0,d,y, yerr),bounds=bounds)
        return soln



def calc_likeli_param(x:tuple[list[float],list[float]],y:list[float],yerr:list[float],d:float,k:float, m:float, gamma:float)-> list[float]:
    t_vec, v_vec = x
    df = pd.DataFrame({'delta_t': t_vec, 'flux': y, 'freq': v_vec})
    R0,N0,tp =calc_init_guess_n_R(df,'flux', alpha,d,eps_b) #take the initial guess from 1.43GHz
    # R0,m,RHO0, k,gamma
    init = np.array([np.log10(R0),m,N0,k,gamma]) # initial parameters
    # init = np.array([np.log10(R0),m,9,k,gamma]) # initial parameters

    print("init_guess", init)
    bnds = ((14,17),(0.67,1),(1,17),(-5,0),(2,4)) # parameters bounds
    soln = max_likeli(initial=init,bounds=bnds,x=x,t0=tp*DAYS,d=d,y=y,yerr=yerr) # solving
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

def run_mcmc_get_sampler(x:tuple[list[float],list[float]],y:list[float],yerr:list[float],tp:float,d:float,nwalkers:int,ndim:int,nsteps:int,solnx:list[float])->EnsembleSampler:
    # Running the MCMC process
    pos = solnx + 1e-4 * np.random.randn(nwalkers,ndim) # initial position of the chain
    # Running in parallel
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, tp*DAYS, d,y,yerr),pool=pool) # setting posterior sampler
        start = time.time()
        sampler.run_mcmc(initial_state=pos, nsteps=nsteps, progress=True) # run mcmc sampler
        end = time.time()
        multi_time = end - start
        print("Serial took {0:.1f} seconds".format(multi_time))
    return sampler

def extract_mcmc_params(sampler: EnsembleSampler)->list[float]:
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    R0_mcmc = np.median(flat_samples[:,0])
    m_mcmc = np.median(flat_samples[:,1])
    N0_mcmc = np.median(flat_samples[:,2])
    k_mcmc = np.median(flat_samples[:,3])
    gamma_mcmc = np.median(flat_samples[:,4])
    print(R0_mcmc,m_mcmc,N0_mcmc,k_mcmc,gamma_mcmc)
    return [R0_mcmc,m_mcmc,N0_mcmc,k_mcmc,gamma_mcmc]

#error calculations and showing

def calc_params_err(sampler: EnsembleSampler, mcmc_param: list):
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
                     y:list[float], yerr:list[float],xlim:tuple[float, float],
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
        temp_y_fit = mult_fitted_function(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_t_fit/DAYS,temp_y_fit,color=plt.gca().lines[-1].get_color())

        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = mult_fitted_function(temp_x_fit,tp*DAYS,d,*sample)
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
        temp_y_fit = mult_fitted_function(temp_x_fit,tp*DAYS,d,*temp_theta_fit)
        ax.plot(temp_nu_fit/Ghz,temp_y_fit,color=plt.gca().lines[-1].get_color())
        
        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            tempf_fit = mult_fitted_function(temp_x_fit,tp*DAYS,d,*sample)
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
    m= (-4*b+4*a*gamma+20*a+2*gamma+10)/(62+10*gamma)
    k = 6-(4*a-2)/m
    return m,k

# mass loss calculation - correct only when k=-2
def calc_mass_loss(mcmc_param: list, tp: float, t_vec: list):
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
    t_epocs = np.unique(np.asarray(df["delta_t"].values))
    v_epocs = np.unique(np.asarray(df["delta_t"].values))
    print("time epocs (DAYS): {}".format(t_epocs))
    print("freq epocs (Ghz): {}".format(v_epocs))

def remove_epoc(df:pd.DataFrame, epoch: float,col_name: str)->pd.DataFrame:
    new_df = df[df[col_name] != epoch]
    return new_df

def single_epoch(df:pd.DataFrame, epoch: float,col_name: str)->pd.DataFrame:
    new_df = df[df[col_name] == epoch]
    return new_df

# func for running the program all together. 
def run_prog(csv_path: str,dist_to_sn: float,init_k: float, init_m: float, init_gamma:float, nwalkers: int,nsteps: int, spectra: bool, light_curve: bool, slegend,lclegend,xlim=None,ylim=None):
    # dist_to_sn  - decide wheather in Mpc or not
    df= csv_file_to_dfs(csv_path)
    df["err"]=[np.sqrt((0.1*f)**2+e**2) for f,e in zip(df["flux"].values,df["flux_err"].values)]

    v_vec = df['freq'].values*Ghz
    t_vec =df['delta_t'].values*DAYS
    
    # y = df['flux'].values*(1e-3) #micro to mili 
    # yerr = df['flux_err'].values*(1e-3) #micro to mili

    y = df['flux'].values 
    # yerr =  df['flux_err'].values 
    yerr = df['flux_err'].values #micro to mili
    # yerr =  df['err'].values
    x = (t_vec, v_vec)
    d=dist_to_sn
    tp,fp,vp = find_flux_time_peak_value(df,'flux')
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