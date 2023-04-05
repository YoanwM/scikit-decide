import warnings
import argparse
from argparse import Action
from argparse import Action
from enum import Enum
from time import sleep
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Dict
import matplotlib.pyplot as plt
import pandas as pd
from flightplanning_utils import (
    flying,
    plot_trajectory,
)
from IPython.display import clear_output
from openap.extra.aero import distance, mach2tas, kts, ft
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
    pos: Tuple[int, int]

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
                     self.mass,
                     self.alt,
                     self.time))

    def __eq__(self, other):
        return (self.pos == other.pos and 
                self.mass == other.mass and 
                self.alt == other.alt and 
                self.time == other.time 
                )

    def __ne__(self, other):
        return (self.pos != other.pos or 
                self.mass != other.mass or 
                self.alt != other.alt or 
                self.time != other.time 
                )

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.pos} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"

class Action(Enum):
    """
    Action that can be perform by the aircraft (in a graph)
    """
    up = -1
    straight = 0
    down = 1

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
    T_event = Action  # Type of events
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
        nb_points_forward: int=41,
        nb_points_lateral: int=11,
        fuel_loaded: float = None
    ):
        """
        Initialisation of a instance of flight planning that will be computed

        Args:
            origin (Union[str, tuple]): 
                ICAO code of the airport, or a tuple (lat,lon), of the origin of the flight plan
            destination (Union[str, tuple]): 
                ICAO code of the airport, or a tuple (lat,lon), of the destination of the flight plan
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
            fuel_loaded (float, optional): 
                Fuel loaded in the aricraft for the flight plan. Defaults to 0.0.
        """
        
        # Initialisation of the origin and the destination    
        if isinstance(origin, str): # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]
        else: # Origin is geographic coordinates 
            self.lat1, self.lon1 = origin

        if isinstance(destination, str): # Origin is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]
        else: # Origin is geographic coordinates 
            self.lat2, self.lon2 = destination
        
        
        # Initialisation of the objective, the constraints and the wind interpolator
        self.objective = objective
        self.constraints = constraints
        self.wind_ds = None
        if wind_interpolator:
            self.wind_ds = wind_interpolator

        # Build network between top of climb and destination airport
        self.np: int = nb_points_forward
        self.nc: int = nb_points_lateral
        self.network = self.get_network(
            LatLon(self.lat1, self.lon1),
            LatLon(self.lat2, self.lon2),
            self.np,
            self.nc,
        )

        # Retrieve the aircraft datas in openap library
        
        self.ac = aircraft(actype)
        
        # Initialisation of the flight plan, with the iniatial state
        assert(fuel_loaded <= self.ac["limits"]['MFC'])
        if not fuel_loaded :
            self.start = State(
                pd.DataFrame(
                    [
                        {
                            "ts": 0, # time of the flight plan, initialised to 0
                            "lat": self.lat1, # latitude of the origin of the flight plan
                            "lon": self.lon1, # longitude of the origin of the flight plan
                            "mass": self.ac["limits"]["MTOW"], # Initialisation of the mass of the aircraft, here with all fuel loaded we reached Maximum TakeOff Weight
                            "mach": self.ac["cruise"]["mach"], # Initialisation of the speed of the aircraft, in mach
                            "fuel": 0.0, # Fuel consummed initialisation 
                            "alt": self.ac["cruise"]["height"], # Altitude of the origin, in meters
                        }
                    ]
                ),
                (0, self.nc // 2),
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
                            "mach": self.ac["cruise"]["mach"],
                            "fuel": 0.0,
                            "alt": self.ac["cruise"]["height"],
                        }
                    ]
                ),
                (0, self.nc // 2),
            )
        print(f"Start : {self.start}")
        # Definition of the fuel consumption function 
        self.fuel_flow = FuelFlow(actype).enroute
        
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """
        Compute the next state from:
          - memory: the current state
          - action: the action to take
        """
        
        trajectory = memory.trajectory.copy()

        # Set intermediate destination point
        next_x, next_y = memory.pos

        next_x += 1

        if action == Action.up:
            next_y += 1
        if action == Action.down:
            next_y -= 1

        # Aircraft stays on the network
        if next_x >= self.np or next_y < 0 or next_y >= self.nc:
            return memory

        # Concatenate the two trajectories

        to_lat = self.network[next_x][next_y].lat
        to_lon = self.network[next_x][next_y].lon
        trajectory = flying(
            trajectory.tail(1), (to_lat, to_lon), self.wind_ds, self.fuel_flow
        )

        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            (next_x, next_y),
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
        return ImplicitSpace(lambda x: x.pos[0]==self.np - 1)

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
        return state.pos[0] == self.np - 1

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        x, y = memory.pos

        space = []
        if x < self.np - 1:
            space.append(Action.straight)
            if y + 1 < self.nc:
                space.append(Action.up)
            if y > 0:
                space.append(Action.down)

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace(Action)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return MultiDiscreteSpace([self.np, self.nc])

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

        if objective == "fuel" : #ajouter bearing
            
            tas = mach2tas(s.trajectory.iloc[-1]["mach"], s.trajectory.iloc[-1]["alt"])
            cost = (distance_to_goal/(tas*kts)) * self.fuel_flow(s.trajectory.iloc[-1]["mass"],
                                                                 tas * kts,
                                                                 s.trajectory.iloc[-1]["alt"] * ft)
                                             
        if objective == "time" :
            cost = distance_to_goal/s.trajectory.iloc[-1]["mach"]

        return Value(cost=cost)

    def get_network(self, p0: LatLon, p1: LatLon, np: int, nc: int):
        np2 = np // 2
        nc2 = nc // 2

        distp = 10 * p0.distanceTo(p1) / np / nc  # meters

        pt = [[None for j in range(nc)] for i in range(np)]

        # set boundaries
        for j in range(nc):
            pt[0][j] = p0
            pt[np - 1][j] = p1

        # direct path between p0 and p1
        for i in range(1, np - 1):
            bearing = pt[i - 1][nc2].initialBearingTo(p1)
            total_distance = pt[i - 1][nc2].distanceTo(pt[np - 1][nc2])
            pt[i][nc2] = pt[i - 1][nc2].destination(total_distance / (np - i), bearing)

        bearing = pt[np2 - 1][nc2].initialBearingTo(pt[np2 + 1][nc2])
        pt[np2][nc - 1] = pt[np2][nc2].destination(distp * nc2, bearing + 90)
        pt[np2][0] = pt[np2][nc2].destination(distp * nc2, bearing - 90)

        for j in range(1, nc2 + 1):
            # +j (left)
            bearing = pt[np2][nc2 + j - 1].initialBearingTo(pt[np2][nc - 1])
            total_distance = pt[np2][nc2 + j - 1].distanceTo(pt[np2][nc - 1])
            pt[np2][nc2 + j] = pt[np2][nc2 + j - 1].destination(
                total_distance / (nc2 - j + 1), bearing
            )
            # -j (right)
            bearing = pt[np2][nc2 - j + 1].initialBearingTo(pt[np2][0])
            total_distance = pt[np2][nc2 - j + 1].distanceTo(pt[np2][0])
            pt[np2][nc2 - j] = pt[np2][nc2 - j + 1].destination(
                total_distance / (nc2 - j + 1), bearing
            )
            for i in range(1, np2):
                # first halp (p0 to np2)
                bearing = pt[i - 1][nc2 + j].initialBearingTo(pt[np2][nc2 + j])
                total_distance = pt[i - 1][nc2 + j].distanceTo(pt[np2][nc2 + j])
                pt[i][nc2 + j] = pt[i - 1][nc2 + j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
                bearing = pt[i - 1][nc2 - j].initialBearingTo(pt[np2][nc2 - j])
                total_distance = pt[i - 1][nc2 - j].distanceTo(pt[np2][nc2 - j])
                pt[i][nc2 - j] = pt[i - 1][nc2 - j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
                # second half (np2 to p1)
                bearing = pt[np2 + i - 1][nc2 + j].initialBearingTo(pt[np - 1][nc2 + j])
                total_distance = pt[np2 + i - 1][nc2 + j].distanceTo(
                    pt[np - 1][nc2 + j]
                )
                pt[np2 + i][nc2 + j] = pt[np2 + i - 1][nc2 + j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
                bearing = pt[np2 + i - 1][nc2 - j].initialBearingTo(pt[np - 1][nc2 - j])
                total_distance = pt[np2 + i - 1][nc2 - j].distanceTo(
                    pt[np - 1][nc2 - j]
                )
                pt[np2 + i][nc2 - j] = pt[np2 + i - 1][nc2 - j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
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
    
    def solve(self, domain_factory, max_steps:int=100, debug:bool=False, make_img:bool = True):
        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=debug)
        self.solve_with(solver,domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation = self.reset()

        solver.reset()
        clear_output(wait=True)
        figure = self.render(observation)
        plt.savefig("look")

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
            print("policy = ", action)
            print("New state = ", observation.pos)
            if make_img : 
                # update image
                plt.clf()  # clear figure
                clear_output(wait=True)
                figure = self.render(observation)
                plt.savefig(f'step_{i_step}')

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img  :
            plt.savefig("terminal")
            
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

def fuel_optimisation(origin : Union[str, tuple], destination : Union[str, tuple], ac : str, constraints : dict, wind_interpolator : GenericWindInterpolator) -> float:
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
    while fuel_remaining and step < 20 and not small_diff:
        domain_factory = lambda: FlightPlanningDomain(origin, 
                                                destination, 
                                                ac, 
                                                constraints=constraints,
                                                wind_interpolator=wind_interpolator, 
                                                objective="distance",
                                                nb_points_forward=41,
                                                nb_points_lateral=11,
                                                fuel_loaded = constraints["fuel"]
                                                )
        domain = domain_factory()
        
        fuel_prec = constraints["fuel"]
        constraints["fuel"],fuel_remaining = domain.simple_fuel_loop(domain_factory)
        step += 1
        small_diff = ((fuel_prec - constraints['fuel']) <= (1/1000))
    
    return constraints["fuel"]


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
            timeConstraint = (float(args.timeConstraintStart),0.0)
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
        print(weather)
    else :
        weather = None

    diversion = args.diversion
    fuel_loop = args.fuelLoop
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
    
    # Doing the fuel loop if requested
    
    if fuel_loop :
        fuel_loaded = fuel_optimisation(origin, destination, ac, constraints, wind_interpolator)
        # Adding fuel reserve (we can't put more fuel than maxFuel)
        fuel_loaded = min(1.1 * fuel_loaded,maxFuel)
    else :
        fuel_loaded = maxFuel
        
    
    constraints["fuel"] = 0.97* fuel_loaded # Update of the maximum fuel there is to be used
    print(f'\n*********************\nFuel loaded : {fuel_loaded}, maximum fuel : {constraints["fuel"]}\n*********************\n')
    
    # Creating the domain 
    domain_factory = lambda: FlightPlanningDomain(origin, 
                                                  destination, 
                                                  ac, 
                                                  constraints=constraints,
                                                  wind_interpolator=wind_interpolator, 
                                                  objective=objective,
                                                  nb_points_forward=41,
                                                  nb_points_lateral=11,
                                                  fuel_loaded=fuel_loaded
                                                 )
    domain = domain_factory()
    
    solvers = match_solvers(domain=domain)

    domain.solve(domain_factory)  
    
    