import warnings
import os, sys, argparse
from argparse import Action
from datetime import datetime, timedelta
from enum import Enum
from time import sleep
from typing import Any, List, NamedTuple, Optional, Tuple, Union
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from flightplanning_utils import (
    flying,
    plot_trajectory,
    plot_network
)
from IPython.display import clear_output
from openap.extra.aero import distance
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft as openap_aircraft
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
          objective = "fuel", 
          timeConstraint = None,
          weather={"year":'2023',
                   "month":'01',
                   "day":'13',
                   "forecast":'nowcast'},
          num = None):


    maxFuel = openap_aircraft(aircraft)['limits']['MFC']
    constraints = {'time' : timeConstraint, # Aircraft should arrive before a given time (or in a given window)
                   'fuel' : 0.97*maxFuel} # Aircraft should arrive with some fuel remaining  
    print(constraints)

    
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

    objective = "distance"   
   
    domain_factory = lambda: FlightPlanningDomain(origin, 
                                                  destination, 
                                                  aircraft, 
                                                  constraints=constraints,
                                                  wind_interpolator=wind_interpolator, 
                                                  objective=objective,
                                                  nb_points_forward=41,
                                                  nb_points_lateral=11,
                                                  )
    domain = domain_factory()
    plot_network(domain)
    match_solvers(domain=domain)
    
    """if objective == "fuel" :
        def heuristic(d,s):
            return d.heuristic(s)
    elif objective == "time":
        def heuristic(d,s):
            return d.heuristic(s)
    elif objective == "distance":
        def heuristic(d,s):
            return d.heuristic(s)"""
        
    solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=debug)
    #solver = LazyAstar(heuristic=lambda d, s: d.heuristic(s))
    FlightPlanningDomain.solve_with(solver, domain_factory)
    pause_between_steps = None
    max_steps = 100
    observation = domain.reset()
    # Initialize image
    #figure = domain.render(observation)
    #plt.savefig('init.png')
    #plt.show()
    solver.reset()
    plt.clf()  # clear figure
    clear_output(wait=True)
    figure = domain.render(observation)
    plt.savefig("look")
    #solver._solve_domain(lambda: domain)
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
    if num is None :
        plt.savefig("terminal")
    else : 
        plt.savefig("terminal"+num)
    # goal reached?
    is_goal_reached = domain.is_goal(observation)
    terminal_state_constraints = domain._get_terminal_state_time_fuel(observation)
    if is_goal_reached :
        if constraints['time'] is not None :
            if constraints['time'][1] >= terminal_state_constraints['time'] :
                if constraints['fuel'] >= terminal_state_constraints['fuel'] :
                    print(f"Goal reached in {i_step} steps!")
                else : 
                    print(f"Goal reached in {i_step} steps, but there is not enough fuel remaining!")
            else : 
                print(f"Goal reached in {i_step} steps, but not in the good timelapse!")
        else :
            if constraints['fuel'] >= terminal_state_constraints['fuel'] :
                print(f"Goal reached in {i_step} steps!")
            else : 
                print(f"Goal reached in {i_step} steps, but there is not enough fuel remaining!")
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
    parser.add_argument('-obj', '--objective', help='Objective for the flight (time, fuel, distance)', type=str)
    parser.add_argument('-tcs', '--timeConstraintStart', help='Start Time constraint for the flight. The flight should arrive after that time')
    parser.add_argument('-tce', '--timeConstraintEnd', help='End Time constraint for the flight. The flight should arrive before that time')
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

    if args.objective :
        objective = args.objective
    else : 
        objective = "fuel"
        
    if args.timeConstraintStart :
        if args.timeConstraintEnd :
            timeConstraint = (float(args.timeConstraintStart),float(args.timeConstraintEnd))
        else : 
            timeConstraint = (float(args.timeConstraintStart),0.0)
    else:
        if args.timeConstraintEnd :
            timeConstraint = (0.0,float(args.timeConstraintEnd))
        else :
            timeConstraint = None
    

    solve(filename, debug = True, 
          destination=destination, 
          origin=origin, 
          aircraft=aircraft, 
          objective=objective, 
          timeConstraint=timeConstraint)
    #solve(filename="instance3.grib")