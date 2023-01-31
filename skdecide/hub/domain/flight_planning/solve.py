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
    WeatherRetrieverFromEcmwf,
    WindInterpolator,
    flying,
    plot_trajectory,
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



 
def solve(filename = None, 
          origin = "LFPG", 
          destination = "WSSS", 
          aircraft="A388", 
          debug = False, 
          objective = "fuel", 
          timeConstraint = None):

    wind_interpolator: WindInterpolator = None
    file = None
    if filename is not None :
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            "instances/%s" % filename))
    if file:
        
        wind_interpolator = WindInterpolator(file)
    if wind_interpolator:
        wind_dataset = wind_interpolator.get_dataset()
        wind_dataset.u.values -= 60
        wind_dataset.v.values += 60
        axes = wind_interpolator.plot_wind(alt=35000.0, t=[0], plot_wind=True)
        plt.savefig('wind')
        # plt.show()
    maxFuel = openap_aircraft(aircraft)['limits']['MFC']
    constraints = {'time' : timeConstraint, # Aircraft should arrive before a given time (or in a given window)
                   'fuel' : 0.97*maxFuel} # Aircraft should arrive with some fuel remaining  
    print(constraints)
    
    
    
    
    
    domain_factory = lambda: FlightPlanningDomain(origin, 
                                                  destination, 
                                                  aircraft, 
                                                  constraints=constraints,
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
    terminal_state_constraints = domain._get_terminal_state_(observation)
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