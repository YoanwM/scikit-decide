import warnings
import argparse
from argparse import Action
from enum import Enum
from time import sleep
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Dict
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from flightplanning_utils import (
    plot_trajectory,
    plot_network,
    plot_altitude
)
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from IPython.display import clear_output
from openap.extra.aero import distance, mach2tas, kts, ft, atmos, latlon
from openap.extra.aero import bearing as aero_bearing
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.solver.astar import Astar
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide import ImplicitSpace
from skdecide.utils import match_solvers

from weather_interpolator.weather_tools.get_weather_noaa import get_weather_matrix
from weather_interpolator.weather_tools.interpolator.GenericInterpolator import GenericWindInterpolator


class State:
    """
    Definition of a aircraft state during the flight plan
    """
    trajectory: pd.DataFrame
    pos: Tuple[int, int, int]

    def __init__(self, trajectory, pos):
        self.trajectory = trajectory
        self.pos = pos
        if trajectory is not None :
            self.mass = trajectory.iloc[-1]['mass']
            self.alt = trajectory.iloc[-1]['alt']
            self.time = trajectory.iloc[-1]['ts']
        else : 
            self.mass = None
            self.alt = None
            self.time = None
    
    def __hash__(self):
        return hash((self.pos,
                     int(self.mass),
                     self.alt,
                     int(self.time)))

    def __eq__(self, other):
        return (self.pos == other.pos and 
                int(self.mass) == int(other.mass) and 
                self.alt == other.alt and 
                int(self.time) == int(other.time) 
                )

    def __ne__(self, other):
        return (self.pos != other.pos or 
                int(self.mass) != int(other.mass) or 
                self.alt != other.alt or 
                int(self.time) != int(other.time) 
                )

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.pos} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"

class H_Action(Enum):
    """
    Horizontal action that can be perform by the aircraft 
    """
    up = -1
    straight = 0
    down = 1

class V_Action(Enum):
    """
    Vertical action that can be perform by the aircraft
    """
    climb = 1
    cruise = 0
    descent = -1

class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent



class FlightPlanningDomain(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Tuple[H_Action,V_Action]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent
    
    
    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        wind_interpolator: GenericWindInterpolator = None,
        objective: str = "fuel",
        constraints = None,
        nb_forward_points: int=41,
        nb_lateral_points: int=11,
        nb_vertical_points: int=None,
        fuel_loaded: float = None,
        fuel_loop: bool = False,
        mach: float = None
    ):
        """
        Initialisation of a instance of flight planning that will be computed

        Args:
            origin (Union[str, tuple]): 
                ICAO code of the airport, or a tuple (lat,lon,alt), of the origin of the flight plan
            destination (Union[str, tuple]): 
                ICAO code of the airport, or a tuple (lat,lon,alt), of the destination of the flight plan
            actype (str): 
                Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
            wind_interpolator (GenericWindInterpolator, optional): 
                Wind interpolator for the flight plan. Defaults to None.
            objective (str, optional): 
                Objective of the flight plan, it will also guide the aircraft through the graph with a defined heuristic. It can be fuel, distance or time. Defaults to "fuel".
            constraints (_type_, optional): 
                Constraints dictionnary (keyValues : ['time', 'fuel'] ) to be defined in for the flight plan. Defaults to None.
            nb_points_forward (int, optional): 
                Number of forward nodes in the graph. Defaults to 41.
            nb_points_lateral (int, optional): 
                Number of lateral nodes in the graph. Defaults to 11.
            nb_points_vertical (int, optional):
                Number of vertical nodes in the graph. Defaults to None.
            fuel_loaded (float, optional): 
                Fuel loaded in the aricraft for the flight plan. Defaults to None.
            mach (float, optional):
                Speed of the aircraft calculated for the flight. Defaults to None
        """
        
        # Initialisation of the origin and the destination    
        if isinstance(origin, str): # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1, self.alt1 = ap1["lat"], ap1["lon"], ap1["alt"]
        else: # Origin is geographic coordinates 
            self.lat1, self.lon1, self.alt1 = origin

        if isinstance(destination, str): # Destination is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2, self.alt2 = ap2["lat"], ap2["lon"], ap2["alt"]
        else: # Destination is geographic coordinates 
            self.lat2, self.lon2, self.alt2 = destination
        
        # Retrieve the aircraft datas in openap library and normalizing meters into ft
        self.ac = aircraft(actype)
        self.ac['limits']['ceiling'] /= ft
        self.ac['cruise']['height'] /= ft
        if mach :
            self.mach = mach
        else :
            self.mach = self.ac['cruise']['mach']
        # Initialisation of the objective, the constraints and the wind interpolator
        self.objective = objective
        self.constraints = constraints
        self.wind_ds = None
        if wind_interpolator:
            self.wind_ds = wind_interpolator

        # Build network between top of climb and destination airport
        self.nb_forward_points = nb_forward_points
        self.nb_lateral_points = nb_lateral_points
        if nb_vertical_points :
            self.nb_vertical_points = nb_vertical_points
        else :
            self.nb_vertical_points = int((self.ac['limits']['ceiling'] - self.ac['cruise']['height']) / 1000) + 1
        self.network = self.get_network(
            LatLon(self.lat1, self.lon1, self.alt1),
            LatLon(self.lat2, self.lon2, self.alt2),
            self.nb_forward_points,
            self.nb_lateral_points,
            self.nb_vertical_points
        )
        
        # Initialisation of the flight plan, with the iniatial state
        if fuel_loaded :
            constraints["fuel"] = 0.97* fuel_loaded
            
        if fuel_loop :
            fuel_loaded = fuel_optimisation(origin, destination, ac, constraints, wind_interpolator,mach)
            # Adding fuel reserve (we can't put more fuel than maxFuel)
            fuel_loaded = min(1.1 * fuel_loaded, maxFuel)
        elif fuel_loaded :
            constraints["fuel"] = 0.97* fuel_loaded # Update of the maximum fuel there is to be used
        else :
            fuel_loaded = maxFuel
            
        
        
        assert(fuel_loaded <= self.ac["limits"]['MFC']) # Ensure fuel loaded <= fuel capacity
        if not fuel_loaded :
            self.start = State(
                pd.DataFrame(
                    [
                        {
                            "ts": 0, # time of the flight plan, initialised to 0
                            "lat": self.lat1, # latitude of the origin of the flight plan
                            "lon": self.lon1, # longitude of the origin of the flight plan
                            "mass": self.ac["limits"]["MTOW"], # Initialisation of the mass of the aircraft, here with all fuel loaded we reached Maximum TakeOff Weight
                            "mach": mach, # Initialisation of the speed of the aircraft, in mach
                            "fuel": 0.0, # Fuel consummed initialisation 
                            "alt": self.alt1, # Altitude of the origin, in ft
                        }
                    ]
                ),
                (0, self.nb_lateral_points // 2, 0), # Initial node in the graph
            )
        else : 
            self.start = State( # Same initialisation than above
                pd.DataFrame(
                    [
                        {
                            "ts": 0,
                            "lat": self.lat1,
                            "lon": self.lon1,
                            "mass": self.ac["limits"]["MTOW"] - 0.8*(self.ac["limits"]["MFC"]-fuel_loaded), # Here we compute the weight difference between the fuel loaded and the fuel capacity
                            "mach": mach,
                            "fuel": 0.0,
                            "alt": self.alt1,
                        }
                    ]
                ),
                (0, self.nb_lateral_points // 2, 0),
            )
        print(f"Start : {self.start}")
        # Definition of the fuel consumption function 
        self.fuel_flow = FuelFlow(actype).enroute
    
    # Class functions 
        
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """
        Compute the next state from:
          - memory: the current state
          - action: the action to take
        """
        
        trajectory = memory.trajectory.copy()

        # Set intermediate destination point
        next_x, next_y, next_z = memory.pos

        next_x += 1

        if action[0] == H_Action.up:
            next_y += 1
        if action[0] == H_Action.down:
            next_y -= 1
        if action[1] == V_Action.climb:
            next_z += 1
        if action[1] == V_Action.descent:
            next_z -= 1

        # Aircraft stays on the network
        if next_x >= self.nb_forward_points or next_y < 0 or next_y >= self.nb_lateral_points or next_z < 0 or next_z >= self.nb_vertical_points:
            return memory

        # Concatenate the two trajectories

        to_lat = self.network[next_x][next_y][next_z].lat
        to_lon = self.network[next_x][next_y][next_z].lon
        to_alt = self.network[next_x][next_y][next_z].height
        trajectory = self.flying(
            trajectory.tail(1), (to_lat, to_lon, to_alt))

        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            (next_x, next_y, next_z),
        )
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.

        Set cost to distance travelled between points
        """
        assert memory != next_state, "Next state is the same as the current state" 
        # Have to change -> openAP top ?
        if self.objective == "distance":
            cost = distance(
                memory.trajectory.iloc[-1]["lat"],
                memory.trajectory.iloc[-1]["lon"],
                next_state.trajectory.iloc[-1]["lat"],
                next_state.trajectory.iloc[-1]["lon"],
            )
        elif self.objective == "fuel":
            cost = memory.trajectory.iloc[-1]["mass"]-next_state.trajectory.iloc[-1]["mass"]
        elif self.objective == "time":
            cost = next_state.trajectory.iloc[-1]["ts"]-memory.trajectory.iloc[-1]["ts"] 
        # return Value(cost=1)
        return Value(cost=cost)

    def _get_initial_state_(self) -> D.T_state:
        """
        Get the initial state.

        Set the start position as initial state.
        """
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        """
        Get the domain goals space (finite or infinite set).

        Set the end position as goal.
        """
        return ImplicitSpace(lambda x: x.pos[0]==self.nb_forward_points - 1)

    def _get_terminal_state_time_fuel(self, state:State) -> dict:
        """
        Get the domain terminal state information to compare with the constraints

        Args:
            state (State): terminal state to retrieve the information on fuel and time.

        Returns:
            dict: dictionnary containing both fuel and time information. 
        """
        fuel = 0.0 
        for trajectory in state.trajectory.iloc :
            fuel += trajectory["fuel"]
        return {'time' : state.trajectory.iloc[-1]["ts"],
                'fuel' : fuel}
       
    def _is_terminal(self, state: State) -> D.T_predicate:
        """
        Indicate whether a state is terminal.

        Stop an episode only when goal reached.
        """
        return state.pos[0] == self.nb_forward_points - 1

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        x, y, z = memory.pos

        space = []
        if x < self.nb_forward_points - 1:
            space.append((H_Action.straight,V_Action.cruise))
            if z < self.nb_vertical_points - 1 :
                space.append((H_Action.straight,V_Action.climb))
            if z > 0 :
                space.append((H_Action.straight,V_Action.descent))
            if y + 1 < self.nb_lateral_points:
                space.append((H_Action.up,V_Action.cruise))
                if z < self.nb_vertical_points - 1 :
                    space.append((H_Action.up,V_Action.climb))
                if z > 0 :
                    space.append((H_Action.up,V_Action.descent))
            if y > 0:
                space.append((H_Action.down,V_Action.cruise))
                if z < self.nb_vertical_points - 1 :
                    space.append((H_Action.down,V_Action.climb))
                if z > 0 :
                    space.append((H_Action.down,V_Action.descent))
            

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace((H_Action,V_Action))

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return MultiDiscreteSpace([self.nb_forward_points, self.nb_lateral_points, self.nb_vertical_points])

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
            matplotlib figure
        """

        return plot_trajectory(
            self.lat1, self.lon1, self.lat2, self.lon2, memory.trajectory, self.wind_ds
        )

    def heuristic(self, s: D.T_state, objective : str = None) -> Value[D.T_value]:
        """Heuristic to be used by search algorithms.
            Depending on the objective and constraints. 
        """
        if objective is None :
            objective = self.objective

        lat = s.trajectory.iloc[-1]["lat"]
        lon = s.trajectory.iloc[-1]["lon"]
        # Compute distance in meters
        distance_to_goal = distance(lat, lon, self.lat2, self.lon2)
        
        if objective == "distance" : 
            cost = distance_to_goal

        if objective == "fuel" : 
    
            tas = mach2tas(s.trajectory.iloc[-1]["mach"], s.trajectory.iloc[-1]["alt"])
            cost = (distance_to_goal/(tas*kts)) * self.fuel_flow(s.trajectory.iloc[-1]["mass"],
                                                                 tas * kts,
                                                                 s.trajectory.iloc[-1]["alt"] * ft)
                                             
        if objective == "time" :
            cost = distance_to_goal/s.trajectory.iloc[-1]["mach"]

        return Value(cost=cost)

    def get_network(self, p0: LatLon, p1: LatLon, nb_forward_points: int, nb_lateral_points: int, nb_vertical_points: int):
        
        half_forward_points = nb_forward_points // 2
        half_lateral_points = nb_lateral_points // 2
        possible_altitudes =[((self.ac['cruise']['height']+2000*i) - (self.ac['cruise']['height']%1000)) for i in range(nb_vertical_points)]

        distp = 10 * p0.distanceTo(p1) / nb_forward_points / nb_lateral_points  # meters
        
        descent_dist = 300_000 #meters
        climb_dist = 220_000 #meters
        climbing_ratio = possible_altitudes[0]/climb_dist
        descending_ratio = possible_altitudes[0]/descent_dist
        # Initialisation of the graph matrix 
        # We take 2 more forward_points for the end of climbing and the beginning of the descent
        pt = [[[None for k in range(nb_vertical_points)] for j in range(nb_lateral_points)] for i in range(nb_forward_points)] 

        # set boundaries
        for j in range(nb_lateral_points):
            for k in range (nb_vertical_points):
                pt[0][j][k] = p0
                pt[nb_forward_points - 1][j][k] = p1
                
              
        # set climb phase
        i_initial=1 
        dist=0
        alt = p0.height
        print(f"Ratio {climbing_ratio}")
        while dist < climb_dist :
            local_dist = (pt[i_initial-1][0][0].distanceTo(p1)) / (nb_forward_points - i_initial)
            dist += local_dist
            alt +=  int(local_dist*climbing_ratio)
            print(f"Altitude : {alt} au pt {i_initial}, dist = {dist}")
            for j in range(nb_lateral_points):
                for k in range (nb_vertical_points):
                    bearing = pt[i_initial-1][j][k].initialBearingTo(p1)
                    pt[i_initial][j][k] = pt[i_initial-1][j][k].destination(local_dist, bearing, min(possible_altitudes[0], alt))
            i_initial+=1
                    
        # set last step, descent    Trying backward
        """for j in range(nb_lateral_points):
                for k in range (nb_vertical_points):
                    bearing = pt[nb_forward_points - 1][j][k].initialBearingTo(p0)
                    pt[nb_forward_points - 2][j][k] = pt[nb_forward_points - 1][j][k].destination(descent_dist, bearing,height=possible_altitudes[0])
        
        descent_point = pt[nb_forward_points - 2][0][0] """
        
        
        i_final=1 
        dist=0
        alt = p1.height
        print(f"Ratio {descending_ratio}")
        while dist < descent_dist :
            local_dist = (pt[nb_forward_points - i_final][0][0].distanceTo(p0)) / (nb_forward_points - i_final)
            dist += local_dist
            alt +=  int(local_dist*descending_ratio)
            print(f"Altitude : {alt} au pt {i_final}, dist = {dist}")
            for j in range(nb_lateral_points):
                for k in range (nb_vertical_points):
                    bearing = pt[nb_forward_points - i_final][j][k].initialBearingTo(p0)
                    pt[nb_forward_points - i_final-1][j][k] = pt[nb_forward_points - i_final][j][k].destination(local_dist, bearing, min(possible_altitudes[0], alt))
            i_final+=1        
            
            
        # direct path between end of climbing and beginning of descent 
        for k in range(len(possible_altitudes)):
            for i in range(i_initial, i_final):
                    bearing = pt[i - 1][half_lateral_points][k].initialBearingTo(pt[i_final][0][0])
                    total_distance = pt[i - 1][half_lateral_points][k].distanceTo(pt[nb_forward_points - 2][half_lateral_points][k])
                    pt[i][half_lateral_points][k] = pt[i - 1][half_lateral_points][k].destination(total_distance / (nb_forward_points - i), bearing, height=possible_altitudes[k])

            bearing = pt[half_forward_points - 1][half_lateral_points][k].initialBearingTo(pt[half_forward_points + 1][half_lateral_points][k])
            pt[half_forward_points][nb_lateral_points - 1][k] = pt[half_forward_points][half_lateral_points][k].destination(distp * half_lateral_points, bearing + 90,  height=possible_altitudes[k])
            pt[half_forward_points][0][k] = pt[half_forward_points][half_lateral_points][k].destination(distp * half_lateral_points, bearing - 90, height=possible_altitudes[k])

        for j in range(1, half_lateral_points + 1):
            for k in range(len(possible_altitudes)):
                # +j (left)
                bearing = pt[half_forward_points][half_lateral_points + j - 1][k].initialBearingTo(pt[half_forward_points][nb_lateral_points - 1][k])
                total_distance = pt[half_forward_points][half_lateral_points + j - 1][k].distanceTo(pt[half_forward_points][nb_lateral_points - 1][k])
                pt[half_forward_points][half_lateral_points + j][k] = pt[half_forward_points][half_lateral_points + j - 1][k].destination(total_distance / (half_lateral_points - j + 1), 
                                                                                                                                          bearing, 
                                                                                                                                          height=possible_altitudes[k])
                # -j (right)
                bearing = pt[half_forward_points][half_lateral_points - j + 1][k].initialBearingTo(pt[half_forward_points][0][k])
                total_distance = pt[half_forward_points][half_lateral_points - j + 1][k].distanceTo(pt[half_forward_points][0][k])
                pt[half_forward_points][half_lateral_points - j][k] = pt[half_forward_points][half_lateral_points - j + 1][k].destination(total_distance / (half_lateral_points - j + 1), 
                                                                                                                                          bearing, 
                                                                                                                                          height=possible_altitudes[k])
                for i in range(i_initial, half_forward_points):
                    # first halp (p0 to np2)
                    bearing = pt[i - 1][half_lateral_points + j][k].initialBearingTo(pt[half_forward_points][half_lateral_points + j][k])
                    total_distance = pt[i - 1][half_lateral_points + j][k].distanceTo(pt[half_forward_points][half_lateral_points + j][k])
                    pt[i][half_lateral_points + j][k] = pt[i - 1][half_lateral_points + j][k].destination(total_distance / (half_forward_points - i + 1), 
                                                                                                        bearing, 
                                                                                                        height=possible_altitudes[k])
                   
                    bearing = pt[i - 1][half_lateral_points - j][k].initialBearingTo(pt[half_forward_points][half_lateral_points - j][k])
                    total_distance = pt[i - 1][half_lateral_points - j][k].distanceTo(pt[half_forward_points][half_lateral_points - j][k])
                    pt[i][half_lateral_points - j][k] = pt[i - 1][half_lateral_points - j][k].destination(total_distance / (half_forward_points - i + 1), 
                                                                                                        bearing, 
                                                                                                        height=possible_altitudes[k])
                for i in range(i_final, half_forward_points):     
                    # second half (np2 to p1)
                    bearing = pt[half_forward_points + i - 1][half_lateral_points + j][k].initialBearingTo(pt[nb_forward_points - 1][half_lateral_points + j][k])
                    total_distance = pt[half_forward_points + i - 1][half_lateral_points + j][k].distanceTo(pt[nb_forward_points - 1][half_lateral_points + j][k])
                    pt[half_forward_points + i][half_lateral_points + j][k] = pt[half_forward_points + i - 1][half_lateral_points + j][k].destination(total_distance / (half_forward_points - i + 1), 
                                                                                                                                                    bearing, 
                                                                                                                                                    height=possible_altitudes[k])
                    
                    bearing = pt[half_forward_points + i - 1][half_lateral_points - j][k].initialBearingTo(pt[nb_forward_points - 1][half_lateral_points - j][k])
                    total_distance = pt[half_forward_points + i - 1][half_lateral_points - j][k].distanceTo(pt[nb_forward_points - 1][half_lateral_points - j][k])
                    pt[half_forward_points + i][half_lateral_points - j][k] = pt[half_forward_points + i - 1][half_lateral_points - j][k].destination(total_distance / (half_forward_points - i + 1), 
                                                                                                                                                    bearing, 
                                                                                                                                                    height=possible_altitudes[k])
                
        return pt

    def simple_fuel_loop(self, domain_factory,max_steps:int=100):

        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=False)
        self.solve_with(solver,domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation = self.reset()

        solver.reset()
        clear_output(wait=True)

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):
            
            if pause_between_steps is not None:
                sleep(pause_between_steps)

            # choose action according to solver
            action = solver.sample_action(observation)
            
            # get corresponding action
            outcome = self.step(action)
            observation = outcome.observation
            
            if self.is_terminal(observation):
                break   

        # Retrieve fuel minimum fuel for the flight
        fuel = self._get_terminal_state_time_fuel(observation)['fuel']
        # Evaluate if there is a fuel constraint violation
        enough_fuel = (fuel <= self.constraints["fuel"])
        
        solver._cleanup()
        print(f'fuel : {fuel}, fuel loaded : {self.constraints["fuel"]}')
        return (fuel,enough_fuel)
    
    def simple_mach_loop(self, domain_factory,max_steps:int=100):

        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=False)
        self.solve_with(solver,domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation = self.reset()

        solver.reset()
        clear_output(wait=True)

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):
            
            if pause_between_steps is not None:
                sleep(pause_between_steps)

            # choose action according to solver
            action = solver.sample_action(observation)
            
            # get corresponding action
            outcome = self.step(action)
            observation = outcome.observation
            
            if self.is_terminal(observation):
                break   

        # Retrieve fuel minimum fuel for the flight
        time = self._get_terminal_state_time_fuel(observation)['time']
        # Evaluate if there is a fuel constraint violation
        to_fast = (time <= self.constraints["time"][0])
        to_slow = (time >= self.constraints["time"][1])
        solver._cleanup()
        print(f'Time : {time}, Time constraints : {self.constraints["time"]}, mach: {self.mach}')
        return (to_fast,to_slow)
    
    def solve(self, domain_factory, max_steps:int=100, debug:bool=False, make_img:bool = False):
        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=debug)
        self.solve_with(solver,domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation = self.reset()
        
        solver.reset()
        clear_output(wait=True)

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):
            
            if pause_between_steps is not None:
                sleep(pause_between_steps)

            # choose action according to solver
            action = solver.sample_action(observation)
            
            # get corresponding action
            outcome = self.step(action)
            observation = outcome.observation
            
            print('step ', i_step)
            print("policy = ", action[0], action[1])
            print("New state = ", observation.pos)
            print("Alt = ", observation.alt)
            if make_img : 
                # update image
                plt.clf()  # clear figure
                clear_output(wait=True)
                figure=self.render(observation)
                #plt.savefig(f'step_{i_step}')

            # final state reached?
            if self.is_terminal(observation): 
                break
        if make_img  :
            plt.savefig("terminal")
            plt.clf()
            clear_output(wait=True)
            figure=plot_altitude(observation.trajectory)
            plt.savefig("altitude")
        # goal reached?
        is_goal_reached = self.is_goal(observation)
        terminal_state_constraints = self._get_terminal_state_time_fuel(observation)
        print(terminal_state_constraints)
        if is_goal_reached :
            if self.constraints['time'] is not None :
                if self.constraints['time'][1] >= terminal_state_constraints['time'] :
                    if self.constraints['fuel'] >= terminal_state_constraints['fuel'] :
                        print(f"Goal reached after {i_step} steps!")
                    else : 
                        print(f"Goal reached after {i_step} steps, but there is not enough fuel remaining!")
                else : 
                    print(f"Goal reached after {i_step} steps, but not in the good timelapse!")
            else :
                if self.constraints['fuel'] >= terminal_state_constraints['fuel'] :
                    print(f"Goal reached after {i_step} steps!")
                else : 
                    print(f"Goal reached after {i_step} steps, but there is not enough fuel remaining!")
        else:
            print(f"Goal not reached after {i_step} steps!")
        solver._cleanup()       
    
    def flying(self, from_: pd.DataFrame, to_: Tuple[float, float, int], dest_: Tuple[float, float, int] = None) -> pd.DataFrame:
        """Compute the trajectory of a flying object from a given point to a given point

        Args:
            from_ (pd.DataFrame): the trajectory of the object so far
            to_ (Tuple[float, float]): the destination of the object

        Returns:
            pd.DataFrame: the final trajectory of the object
        """
        pos = from_.to_dict("records")[0]
        
        alt = to_[2]
        dist_ = distance(pos["lat"], pos["lon"], to_[0], to_[1], alt)
        data = []
        epsilon = 100
        dt = 600
        dist = dist_
        loop = 0
        while dist > epsilon:  
            bearing = aero_bearing(pos["lat"], pos["lon"], to_[0], to_[1])
            p, _, _ = atmos(alt*ft)
            isobaric = p / 100
            we, wn = 0, 0
            if self.wind_ds:
                time = pos["ts"]
                wind_ms = self.wind_ds.interpol_wind_classic(
                    lat=pos["lat"],
                    longi=pos["lon"],
                    alt=alt,
                    t=time
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

            wdir = (degrees(atan2(we, wn)) + 180) % 360
            wspd = sqrt(wn * wn + we * we)
            
            tas = mach2tas(pos["mach"], alt*ft)  # 400

            wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

            heading = (360 + bearing - degrees(wca)) % 360

            gsn = tas * cos(radians(heading)) - wn
            gse = tas * sin(radians(heading)) - we

            gs = sqrt(gsn * gsn + gse * gse) # ground speed
            
            if gs*dt > dist :
                # Last step. make sure we go to destination.
                dt = dist/gs
                ll = to_[0], to_[1]
            else:
                brg = degrees(atan2(gse, gsn)) % 360.0
                ll = latlon(pos["lat"], pos["lon"], gs * dt, brg, alt*ft)
            pos["fuel"] = dt * self.fuel_flow(pos["mass"], 
                                    tas / kts, 
                                    alt * ft, 
                                    path_angle=(alt-pos['alt'])/(gs*dt))
            mass = pos["mass"] - pos["fuel"]

            new_row = {
                "ts": pos["ts"] + dt,
                "lat": ll[0],
                "lon": ll[1],
                "mass": mass,
                "mach": pos["mach"],
                "fuel": pos["fuel"],
                "alt": alt, # to be modified
            }

            # New distance to the next 'checkpoint'
            dist = distance(new_row["lat"], new_row["lon"], 
                            to_[0], to_[1], new_row["alt"])
            
            #print("Dist : %f Dist_ : %f " %(dist,dist_))
            if dist < dist_:
                #print("Fuel new_row : %f" %new_row["fuel"])
                data.append(new_row)
                dist_ = dist
                pos = data[-1]
            else:
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            loop += 1

        return pd.DataFrame(data)


def fuel_optimisation(origin : Union[str, tuple], destination : Union[str, tuple], ac : str, constraints : dict, wind_interpolator : GenericWindInterpolator, mach : float) -> float:
    """
    Function to optimise the fuel loaded in the plane, doing multiple fuel loops to approach an optimal

    Args:
        origin (Union[str, tuple]): 
            ICAO code of the departure airport of th flight plan e.g LFPG for Paris-CDG, or a tuple (lat,lon)
        
        destination (Union[str, tuple]): 
            ICAO code of the arrival airport of th flight plan e.g LFBO for Toulouse-Blagnac airport, or a tuple (lat,lon)
        
        ac (str): 
            Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
        
        constraints (dict): 
            Constraints that will be defined for the flight plan 
        
        wind_interpolator (GenericWindInterpolator): 
            Define the wind interpolator to use wind informations for the flight plan
            
        fuel_loaded (float):
            Fuel loaded in the plane for the flight 
    Returns:
        float: 
            Return the quantity of fuel to be loaded in the plane for the flight
    """
     
    fuel_remaining = True
    small_diff = False
    step = 0 
    new_fuel = constraints["fuel"]
    while (not small_diff)  :
        fuel_domain_factory = lambda: FlightPlanningDomain(origin, 
                                                destination, 
                                                ac, 
                                                constraints=constraints,
                                                wind_interpolator=wind_interpolator, 
                                                objective="time",
                                                nb_forward_points=41,
                                                nb_lateral_points=11,
                                                nb_vertical_points=1,
                                                fuel_loaded = new_fuel,
                                                mach = mach
                                                )
        fuel_domain = fuel_domain_factory()
        
        fuel_prec = new_fuel
        new_fuel,fuel_remaining = fuel_domain.simple_fuel_loop(fuel_domain_factory)
        step += 1
        small_diff = ((fuel_prec - new_fuel) <= (1/1000))
    
    return new_fuel

def mach_optimisation(origin : Union[str, tuple], destination : Union[str, tuple], ac : str, constraints : dict, wind_interpolator : GenericWindInterpolator) -> float:
    """
    Function to optimise the fuel loaded in the plane, doing multiple fuel loops to approach an optimal

    Args:
        origin (Union[str, tuple]): 
            ICAO code of the departure airport of th flight plan e.g LFPG for Paris-CDG, or a tuple (lat,lon)
        
        destination (Union[str, tuple]): 
            ICAO code of the arrival airport of th flight plan e.g LFBO for Toulouse-Blagnac airport, or a tuple (lat,lon)
        
        ac (str): 
            Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
        
        constraints (dict): 
            Constraints that will be defined for the flight plan 
        
        wind_interpolator (GenericWindInterpolator): 
            Define the wind interpolator to use wind informations for the flight plan
            
        fuel_loaded (float):
            Fuel loaded in the plane for the flight 
    Returns:
        float: 
            Return mach average speed to complete the flight in the good time window
    """
     
    fuel_remaining = True
    small_diff = False
    step = 0
    mach = aircraft(ac)['cruise']['mach']
    new_fuel = constraints["fuel"]
    domain_factory = lambda: FlightPlanningDomain(origin, 
                                                destination, 
                                                ac, 
                                                constraints=constraints,
                                                wind_interpolator=wind_interpolator, 
                                                objective="time",
                                                nb_forward_points=41,
                                                nb_lateral_points=11,
                                                fuel_loaded = new_fuel,
                                                mach = mach
                                                )
    domain = domain_factory()
    too_fast, too_slow = domain.simple_mach_loop(domain_factory)
    step = 0
    while ( (too_fast or too_slow) and mach <= aircraft(ac)['limits']['MMO'] and mach >= (aircraft(ac)['cruise']['mach']-0.05) or step > 10) :
        step += 1
        if too_fast :
            mach -= 0.01
        elif too_slow :
            mach += 0.01
            
        domain_factory = lambda: FlightPlanningDomain(origin, 
                                                destination, 
                                                ac, 
                                                constraints=constraints,
                                                wind_interpolator=wind_interpolator, 
                                                objective="time",
                                                nb_forward_points=41,
                                                nb_lateral_points=11,
                                                fuel_loaded = constraints["fuel"],
                                                mach = mach
                                                )
        domain = domain_factory()
        
        fuel_prec = constraints["fuel"]
        too_fast, too_slow = domain.simple_mach_loop(domain_factory)
        if not fuel_remaining :
            constraints['fuel'] = aircraft(ac)['limits']['MFC']
    
    return mach


if __name__ == "__main__":
    """
    Example of launch : python domain.py -o LFPG -d WSSS -ac A388 -obj fuel -w 2023-01-13
    """
    
    # Definition of command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-deb','--debug', action='store_true')
    parser.add_argument('-d','--destination', help='ICAO code of the destination', type=str)
    parser.add_argument('-o','--origin', help='ICAO code of the origin', type=str)
    parser.add_argument('-ac', '--aircraft', help='ICAO code of the aircraft', type=str)
    parser.add_argument('-obj', '--objective', help='Objective for the flight (time, fuel, distance)', type=str)
    parser.add_argument('-tcs', '--timeConstraintStart', help='Start Time constraint for the flight. The flight should arrive after that time')
    parser.add_argument('-tce', '--timeConstraintEnd', help='End Time constraint for the flight. The flight should arrive before that time')
    parser.add_argument('-w', '--weather', help='Weather day for the weather interpolator, format:YYYYMMDD', type=str)
    parser.add_argument('-div', '--diversion', help='Boolean to put a diversion on the flight plan', action='store_true')
    parser.add_argument('-fl', '--fuelLoop', help='If this option is selected, there will be a first loop to optimise the fuel loaded in the plane',action='store_true')
    parser.add_argument('-img', '--images', help='Saving images from the flight plan e.g. Network points, terminal trajectory...', action='store_true')
    
    
    
    args = parser.parse_args()
    
    # Retrieve arguments 
    
    if args.destination :
        destination = args.destination
    else : 
        destination = 'LFBO' # Toulouse
        destination = "EDDB" # Berlin
        destination = "WSSS" # Singapour

    if args.origin :
        origin = args.origin
    else : 
        origin = 'LFPG' 
    
    debug = args.debug
    
    if args.aircraft :
        ac = args.aircraft
    else : 
        ac = 'A388'

    if args.objective :
        objective = args.objective
    else : 
        objective = "fuel"
        
    if args.timeConstraintStart :
        if args.timeConstraintEnd :
            timeConstraint = (float(args.timeConstraintStart),float(args.timeConstraintEnd))
        else : 
            timeConstraint = (float(args.timeConstraintStart),None)
    else:
        if args.timeConstraintEnd :
            timeConstraint = (0.0,float(args.timeConstraintEnd))
        else :
            timeConstraint = None

    if args.weather :
        if len(args.weather) != 8 :
            weather={"year":'2023',
                   "month":'01',
                   "day":'13',
                   "forecast":'nowcast'}
        else : 
            year = args.weather[0:4]
            month = args.weather[4:6]
            day = args.weather[6:8]
            weather={"year":year,
                    "month":month,
                    "day":day,
                    "forecast":'nowcast'}
    else :
        weather = None

    diversion = args.diversion
    fuel_loop = args.fuelLoop
    make_img = args.images
    
    
    # Define basic constraints 
    
    maxFuel = aircraft(ac)['limits']['MFC']
    constraints = {'time' : timeConstraint, # Aircraft should arrive before a given time (or in a given window)
                   'fuel' : maxFuel} # Aircraft should arrive with some fuel remaining  
    
    # Creating wind interpolator
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
    
    # Doing a speed loop if necessary
    
    if constraints["time"] is not None :
        mach = mach_optimisation(origin, destination, ac, constraints, wind_interpolator)
    else :
        mach = aircraft(ac)['cruise']['mach']   
    # Creating the domain 
    domain_factory = lambda: FlightPlanningDomain(origin, 
                                                  destination, 
                                                  ac, 
                                                  constraints=constraints,
                                                  wind_interpolator=wind_interpolator, 
                                                  objective=objective,
                                                  nb_forward_points=41,
                                                  nb_lateral_points=11,
                                                  fuel_loop=fuel_loop,
                                                  mach = mach
                                                 )
    domain = domain_factory()
    plot_network(domain)
    solvers = match_solvers(domain=domain)

    domain.solve(domain_factory,make_img=make_img,debug=debug)  
    
