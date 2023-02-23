import warnings
import os, sys, argparse
from argparse import Action
from datetime import datetime, timedelta
from enum import Enum
from time import sleep
from typing import Any, NamedTuple, Optional, Tuple, Union
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from flightplanning_utils import (
    WeatherRetrieverFromEcmwf,
    WindInterpolator,
    flying,
    plot_trajectory,
)
from IPython.display import clear_output
from openap.extra.aero import distance
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import match_solvers
from domain import FlightPlanningDomain

from weather_interpolator.weather_tools.get_weather_noaa import get_weather_matrix
from weather_interpolator.weather_tools.interpolator.GenericInterpolator import GenericWindInterpolator

 
def solve(filename = None, 
          origin = "LFPG", 
          destination = "WSSS", 
          aircraft="A388", 
          debug = False, 
          weather={"year":'2023',
                   "month":'01',
                   "day":'13',
                   "forecast":'nowcast'}):
    
    if weather :            
        wind_interpolator = None
        mat = get_weather_matrix(year=weather["year"],
                                month=weather["month"],
                                day=weather["day"],
                                forecast=weather["forecast"],
                                delete_npz_from_local=False,
                                delete_grib_from_local=False)
        wind_interpolator = GenericWindInterpolator(file_npz=mat)
    else : 
        wind_interpolator = None

    objective = "fuel"
    domain_factory = lambda: FlightPlanningDomain(origin, destination, aircraft, 
                                                  wind_interpolator=wind_interpolator, 
                                                  objective=objective)
    domain = domain_factory()

    match_solvers(domain=domain)
    
    if objective == "fuel" or objective == "time":
        heuristic = None
    if objective == "distance":
        def heuristic(d,s):
            return d.heuristic(s)
    """
    print("Starting planning")
    use_lazy_astar = False
    if use_lazy_astar:
        solver = LazyAstar(from_state=domain.get_initial_state(), heuristic=heuristic, verbose=True)
        solver.solve(lambda: domain)
    else:
        solver = Astar(heuristic=heuristic, debug_logs=debug)
        FlightPlanningDomain.solve_with(solver, domain_factory)
        solver.reset()
    """
    solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=False)
    FlightPlanningDomain.solve_with(solver, domain_factory)
    pause_between_steps = None
    max_steps = 100
    observation = domain.reset()
    # Initialize image
    #figure = domain.render(observation)
    #plt.savefig('init.png')
    #plt.show()
    solver.reset()
   
    
    # loop until max_steps or goal is reached
    for i_step in range(1, max_steps + 1):
        if pause_between_steps is not None:
            sleep(pause_between_steps)

        # choose action according to solver
        action = solver.sample_action(observation)
        # get corresponding action
        outcome = domain.step(action)
        observation = outcome.observation
        print('step ', i_step)
        print("policy = ", action)
        print("New state = ", observation.pos)
        # update image
        plt.clf()  # clear figure
        clear_output(wait=True)
        figure = domain.render(observation)
        plt.show()

        # final state reached?
        if domain.is_terminal(observation):
            break
    plt.savefig("terminal")
    # goal reached?
    is_goal_reached = domain.is_goal(observation)
    if is_goal_reached:
        print(f"Goal reached in {i_step} steps!")
    else:
        print(f"Goal not reached after {i_step} steps!")
    solver._cleanup()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',  help='The filename containing the wind data', type=str)
    parser.add_argument('-deb','--debug', action='store_true')
    parser.add_argument('-d','--destination', help='ICAO code of the destination', type=str)
    parser.add_argument('-o','--origin', help='ICAO code of the origin', type=str)
    parser.add_argument('-ac', '--aircraft', help='ICAO code of the aircraft', type=str)
    args = parser.parse_args()
    
    if args.filename :
        filename = args.filename
    else : 
        filename = 'instance3.grib' 
        filename = None
    
    if args.destination :
        destination = args.destination
    else : 
        destination = 'LFBO' 
        destination = "EDDB" # Berlin
        destination = "WSSS" # Singapour

    if args.origin :
        origin = args.origin
    else : 
        origin = 'LFPG' 
    
    debug = args.debug
    
    if args.aircraft :
        aircraft = args.aircraft
    else : 
        aircraft = 'A388'

    

    solve(filename, debug = True, destination=destination, origin=origin, aircraft=aircraft)
    #solve(filename="instance3.grib")