import argparse
import math
from argparse import Action
from enum import Enum
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from time import sleep
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flightplanning_utils import (
    plot_altitude,
    plot_network,
    plot_trajectory,
    plot_trajectory_no_map,
)
from IPython.display import clear_output
from openap.extra.aero import atmos
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, kts, latlon, mach2tas
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon
from weather_interpolator.weather_tools.get_weather_noaa import get_weather_matrix
from weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.solver.astar import Astar
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import match_solvers


class State:
    """
    Definition of a aircraft state during the flight plan
    """

    trajectory: pd.DataFrame
    pos: Tuple[int, int, int]

    def __init__(self, trajectory, pos):
        """Initialisation of a state

        Args:
            trajectory : Trajectory information of the flight
            pos : Current position in the airways graph
        """
        self.trajectory = trajectory
        self.pos = pos
        if trajectory is not None:
            self.mass = trajectory.iloc[-1]["mass"]
            self.alt = trajectory.iloc[-1]["alt"]
            self.time = trajectory.iloc[-1]["ts"]
        else:
            self.mass = None
            self.alt = None
            self.time = None

    def __hash__(self):
        return hash((self.pos, int(self.mass), self.alt, int(self.time)))

    def __eq__(self, other):
        return (
            self.pos == other.pos
            and int(self.mass) == int(other.mass)
            and self.alt == other.alt
            and int(self.time) == int(other.time)
        )

    def __ne__(self, other):
        return (
            self.pos != other.pos
            or int(self.mass) != int(other.mass)
            or self.alt != other.alt
            or int(self.time) != int(other.time)
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


class FlightPlanningDomain(
    DeterministicPlanningDomain, UnrestrictedActions, Renderable
):

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Tuple[H_Action, V_Action]  # Type of events
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
        constraints=None,
        nb_forward_points: int = 41,
        nb_lateral_points: int = 11,
        nb_vertical_points: int = None,
        fuel_loaded: float = None,
        fuel_loop: bool = False,
        climbing_slope: float = None,
        descending_slope: float = None,
        graph_width: str = None,
        res_img_dir: str = None,
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
            climbing_slope (float, optionnal):
                Climbing slope of the aircraft, has to be between 10.0 and 25.0. Defaults to None.
            descending_slope (float, optionnal):
                Descending slope of the aircraft, has to be between 10.0 and 25.0. Defaults to None.
            graph_width (str, optionnal):
                Airways graph width, in ["tiny", "small", "normal", "large", "xlarge"]. Defaults to None
            res_img_dir (str, optionnal):
                Directory in which images will be saved. Defaults to None
        """

        # Initialisation of the origin and the destination
        if isinstance(origin, str):  # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1, self.alt1 = ap1["lat"], ap1["lon"], ap1["alt"]
        else:  # Origin is geographic coordinates
            self.lat1, self.lon1, self.alt1 = origin

        if isinstance(destination, str):  # Destination is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2, self.alt2 = ap2["lat"], ap2["lon"], ap2["alt"]
        else:  # Destination is geographic coordinates
            self.lat2, self.lon2, self.alt2 = destination

        # Retrieve the aircraft datas in openap library and normalizing meters into ft
        self.ac = aircraft(actype)

        self.ac["limits"]["ceiling"] /= ft
        self.ac["cruise"]["height"] /= ft

        self.mach = self.ac["cruise"]["mach"]

        # Initialisation of the objective, the constraints and the wind interpolator
        if ["distance", "fuel", "lazy_fuel", "time", "lazy_time"].__contains__(
            objective
        ):
            self.objective = objective
        else:
            self.objective = None
        self.constraints = constraints
        self.wind_ds = None
        if wind_interpolator:
            self.wind_ds = wind_interpolator

        # Build network between airports
        if graph_width:
            all_graph_width = {
                "tiny": 0.5,
                "small": 0.75,
                "normal": 1.0,
                "large": 1.5,
                "xlarge": 2.0,
            }
            graph_width = all_graph_width[graph_width]
        else:
            graph_width = 1.0

        self.nb_forward_points = nb_forward_points
        self.nb_lateral_points = nb_lateral_points

        if nb_vertical_points:
            self.nb_vertical_points = nb_vertical_points
        else:
            self.nb_vertical_points = (
                int((self.ac["limits"]["ceiling"] - self.ac["cruise"]["height"]) / 1000)
                + 1
            )
        self.network = self.get_network(
            LatLon(self.lat1, self.lon1, self.alt1),
            LatLon(self.lat2, self.lon2, self.alt2),
            self.nb_forward_points,
            self.nb_lateral_points,
            self.nb_vertical_points,
            descending_slope=descending_slope,
            climbing_slope=climbing_slope,
            graph_width=graph_width,
        )

        self.fuel_loaded = fuel_loaded
        # Initialisation of the flight plan, with the iniatial state

        if fuel_loop:
            fuel_loaded = fuel_optimisation(
                origin, destination, ac, constraints, wind_interpolator, self.mach
            )
            # Adding fuel reserve (but we can't put more fuel than maxFuel)
            fuel_loaded = min(1.1 * fuel_loaded, self.ac["limits"]["MFC"])
        elif fuel_loaded:
            self.constraints["fuel"] = (
                0.97 * fuel_loaded
            )  # Update of the maximum fuel there is to be used
        else:
            fuel_loaded = self.ac["limits"]["MFC"]

        self.fuel_loaded = fuel_loaded

        assert (
            fuel_loaded <= self.ac["limits"]["MFC"]
        )  # Ensure fuel loaded <= fuel capacity
        self.start = State(
            pd.DataFrame(
                [
                    {
                        "ts": 0,
                        "lat": self.lat1,
                        "lon": self.lon1,
                        "mass": self.ac["limits"]["MTOW"]
                        - 0.8
                        * (
                            self.ac["limits"]["MFC"] - self.fuel_loaded
                        ),  # Here we compute the weight difference between the fuel loaded and the fuel capacity
                        "mach": self.mach,
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
        self.res_img_dir = res_img_dir

    # Class functions

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """Compute the next state

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform

        Returns:
            D.T_state: The next state
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
        if (
            next_x >= self.nb_forward_points
            or next_y < 0
            or next_y >= self.nb_lateral_points
            or next_z < 0
            or next_z >= self.nb_vertical_points
        ):
            return memory

        # Concatenate the two trajectories

        to_lat = self.network[next_x][next_y][next_z].lat
        to_lon = self.network[next_x][next_y][next_z].lon
        to_alt = self.network[next_x][next_y][next_z].height

        # self.mach = min(self.speed_management(trajectory.tail(1)), self.ac["limits"]["MMO"])
        trajectory = self.flying(trajectory.tail(1), (to_lat, to_lon, to_alt))

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

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform
            next_state (Optional[D.T_state], optional): The next state. Defaults to None.

        Returns:
            Value[D.T_value]: Cost to go from memory to next state
        """
        assert memory != next_state, "Next state is the same as the current state"
        # Have to change -> openAP top ?
        if self.objective == "distance":
            cost = LatLon.distanceTo(
                LatLon(
                    memory.trajectory.iloc[-1]["lat"],
                    memory.trajectory.iloc[-1]["lon"],
                    memory.trajectory.iloc[-1]["alt"],
                ),
                LatLon(
                    next_state.trajectory.iloc[-1]["lat"],
                    next_state.trajectory.iloc[-1]["lon"],
                    next_state.trajectory.iloc[-1]["alt"],
                ),
            )
        elif self.objective == "fuel" or self.objective == "lazy_fuel":
            cost = (
                memory.trajectory.iloc[-1]["mass"]
                - next_state.trajectory.iloc[-1]["mass"]
            )
        elif self.objective == "time" or self.objective == "lazy_time":
            cost = (
                next_state.trajectory.iloc[-1]["ts"] - memory.trajectory.iloc[-1]["ts"]
            )
        else:
            cost = 0
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
        return ImplicitSpace(lambda x: x.pos[0] == self.nb_forward_points - 1)

    def _get_terminal_state_time_fuel(self, state: State) -> dict:
        """
        Get the domain terminal state information to compare with the constraints

        Args:
            state (State): terminal state to retrieve the information on fuel and time.

        Returns:
            dict: dictionnary containing both fuel and time information.
        """
        fuel = 0.0
        for trajectory in state.trajectory.iloc:
            fuel += trajectory["fuel"]
        return {"time": state.trajectory.iloc[-1]["ts"], "fuel": fuel}

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
            space.append((H_Action.straight, V_Action.cruise))
            if z < self.nb_vertical_points - 1:
                space.append((H_Action.straight, V_Action.climb))
            if z > 0:
                space.append((H_Action.straight, V_Action.descent))
            if y + 1 < self.nb_lateral_points:
                space.append((H_Action.up, V_Action.cruise))
                if z < self.nb_vertical_points - 1:
                    space.append((H_Action.up, V_Action.climb))
                if z > 0:
                    space.append((H_Action.up, V_Action.descent))
            if y > 0:
                space.append((H_Action.down, V_Action.cruise))
                if z < self.nb_vertical_points - 1:
                    space.append((H_Action.down, V_Action.climb))
                if z > 0:
                    space.append((H_Action.down, V_Action.descent))

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace((H_Action, V_Action))

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return MultiDiscreteSpace(
            [self.nb_forward_points, self.nb_lateral_points, self.nb_vertical_points]
        )

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
            matplotlib figure
        """
        return plot_trajectory(
            self.lat1,
            self.lon1,
            self.lat2,
            self.lon2,
            memory.trajectory,
            self.wind_ds,
        )

    def heuristic(self, s: D.T_state, objective: str = None) -> Value[D.T_value]:
        """
        Heuristic to be used by search algorithms, depending on the objective and constraints.

        Args:
            s (D.T_state): Actual state
            objective (str, optional): Objective function. Defaults to None.

        Returns:
            Value[D.T_value]: Heuristic value of the state.
        """
        if objective is None:
            objective = self.objective

        pos = s.trajectory.iloc[-1]

        # Compute distance in meters
        distance_to_goal = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"]),
            LatLon(self.lat2, self.lon2, height=self.alt2),
        )
        distance_to_start = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"]),
            LatLon(self.lat1, self.lon1, height=self.alt1),
        )

        if objective == "distance":
            cost = distance_to_goal

        elif objective == "fuel":

            we, wn = 0, 0
            bearing = aero_bearing(pos["lat"], pos["lon"], self.lat2, self.lon2)

            if self.wind_ds:
                time = pos["ts"]
                wind_ms = self.wind_ds.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=time
                )

                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300
            wdir = (degrees(atan2(we, wn)) + 180) % 360
            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(pos["mach"], pos["alt"] * ft)

            wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

            heading = (360 + bearing - degrees(wca)) % 360

            gsn = tas * cos(radians(heading)) - wn
            gse = tas * sin(radians(heading)) - we

            gs = sqrt(gsn * gsn + gse * gse)  # ground speed

            cost = ((distance_to_goal / gs)) * (
                self.fuel_flow(pos["mass"], tas * kts, pos["alt"] * ft)
            )

        elif objective == "time":
            we, wn = 0, 0
            bearing = aero_bearing(pos["lat"], pos["lon"], self.lat2, self.lon2)

            if self.wind_ds:
                time = pos["ts"]
                wind_ms = self.wind_ds.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=time
                )

                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

            wdir = (degrees(atan2(we, wn)) + 180) % 360

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(pos["mach"], pos["alt"] * ft)

            wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

            heading = (360 + bearing - degrees(wca)) % 360

            gsn = tas * cos(radians(heading)) - wn
            gse = tas * sin(radians(heading)) - we

            gs = sqrt(gsn * gsn + gse * gse)  # ground speed

            cost = 1.1 * (distance_to_goal / gs)

        elif objective == "lazy_fuel":
            fuel_consummed = s.trajectory.iloc[0]["mass"] - pos["mass"]
            cost = 1.05 * distance_to_goal * (fuel_consummed / distance_to_start)

        elif objective == "lazy_time":
            cost = (
                1.5
                * distance_to_goal
                * ((pos["ts"] - s.trajectory.iloc[0]["ts"]) / distance_to_start)
            )
        else:
            cost = 0
        return Value(cost=0)  # cost)

    def get_network(
        self,
        p0: LatLon,
        p1: LatLon,
        nb_forward_points: int,
        nb_lateral_points: int,
        nb_vertical_points: int,
        climbing_slope: float = None,
        descending_slope: float = None,
        graph_width: float = None,
    ):
        """
        Creation of the airway graph.

        Args:
            p0 (LatLon): Origin of the flight plan
            p1 (LatLon): Destination of the flight plan
            nb_forward_points (int): Number of forward points in the graph
            nb_lateral_points (int): Number of lateral points in the graph
            nb_vertical_points (int): Number of vertical points in the graph
            climbing_slope (float, optional): Climbing slope of the plane during climbing phase. Defaults to None.
            descending_slope (float, optional):  Descent slope of the plane during descent phase. Defaults to None.
            graph_width (float, optional): Graph width of the graph. Defaults to None.

        Returns:
            A 3D matrix containing for each points its latitude, longitude, altitude between origin & destination.
        """

        cruise_alt_min = 31_000  # maybe useful to change this
        half_forward_points = nb_forward_points // 2
        half_lateral_points = nb_lateral_points // 2
        half_vertical_points = nb_vertical_points // 2

        distp = (
            graph_width * 10 * p0.distanceTo(p1) / nb_forward_points / nb_lateral_points
        )  # meters, around 2.2% of the p0 to p1 distance

        descent_dist = 300_000  # meters
        climb_dist = 220_000  # meters

        total_distance = p0.distanceTo(p1)
        if total_distance < (climb_dist + descent_dist):
            climb_dist = total_distance * (
                (climb_dist / (climb_dist + descent_dist)) - 0.1
            )
            descent_dist = total_distance * (
                descent_dist / (climb_dist + descent_dist) - 0.1
            )
            possible_altitudes = [cruise_alt_min for k in range(nb_vertical_points)]

        else:
            possible_altitudes = [
                (
                    min(
                        self.ac["cruise"]["height"]
                        + 2000 * i
                        - (self.ac["cruise"]["height"] % 1000),
                        self.ac["limits"]["ceiling"],
                    )
                )
                for i in range(nb_vertical_points)
            ]

        if climbing_slope:
            climbing_ratio = climbing_slope
        else:
            climbing_ratio = possible_altitudes[0] / climb_dist
        if descending_slope:
            descending_ratio = descending_slope
        else:
            descending_ratio = possible_altitudes[0] / descent_dist
        # Initialisation of the graph matrix
        pt = [
            [
                [None for k in range(len(possible_altitudes))]
                for j in range(nb_lateral_points)
            ]
            for i in range(nb_forward_points)
        ]

        # set boundaries
        for j in range(nb_lateral_points):
            for k in range(nb_vertical_points):
                pt[0][j][k] = p0
                pt[nb_forward_points - 1][j][k] = p1

        # set climb phase
        i_initial = 1
        dist = 0
        alt = p0.height
        while dist < climb_dist:

            local_dist = (
                pt[i_initial - 1][half_lateral_points][half_vertical_points].distanceTo(
                    p1
                )
            ) / (nb_forward_points - i_initial)
            dist += local_dist
            alt += int(local_dist * climbing_ratio)

            for k in range(nb_vertical_points):
                bearing = pt[i_initial - 1][half_lateral_points][k].initialBearingTo(p1)
                pt[i_initial][half_lateral_points][k] = pt[i_initial - 1][
                    half_lateral_points
                ][k].destination(local_dist, bearing, min(possible_altitudes[0], alt))
            i_initial += 1

        # set last step, descent
        i_final = 1
        dist = 0
        alt = p1.height

        while dist < descent_dist:
            local_dist = (
                pt[nb_forward_points - i_final][half_lateral_points][
                    half_vertical_points
                ].distanceTo(p0)
            ) / (nb_forward_points - i_final)
            dist += local_dist
            alt += int(local_dist * descending_ratio)

            for k in range(nb_vertical_points):
                bearing = pt[nb_forward_points - i_final][half_lateral_points][
                    k
                ].initialBearingTo(p0)
                pt[nb_forward_points - i_final - 1][half_lateral_points][k] = pt[
                    nb_forward_points - i_final
                ][half_lateral_points][k].destination(
                    local_dist, bearing, min(possible_altitudes[0], alt)
                )
            i_final += 1

        # direct path between end of climbing and beginning of descent
        for k in range(nb_vertical_points):
            for i in range(i_initial, nb_forward_points - i_final + 1):
                bearing = pt[i - 1][half_lateral_points][k].initialBearingTo(p1)
                total_distance = pt[i - 1][half_lateral_points][k].distanceTo(
                    pt[nb_forward_points - 2][half_lateral_points][k]
                )
                pt[i][half_lateral_points][k] = pt[i - 1][half_lateral_points][
                    k
                ].destination(
                    total_distance / (nb_forward_points - i),
                    bearing,
                    height=possible_altitudes[k],
                )

            bearing = pt[half_forward_points - 1][half_lateral_points][
                k
            ].initialBearingTo(pt[half_forward_points + 1][half_lateral_points][k])
            pt[half_forward_points][nb_lateral_points - 1][k] = pt[half_forward_points][
                half_lateral_points
            ][k].destination(
                distp * half_lateral_points, bearing + 90, height=possible_altitudes[k]
            )
            pt[half_forward_points][0][k] = pt[half_forward_points][
                half_lateral_points
            ][k].destination(
                distp * half_lateral_points, bearing - 90, height=possible_altitudes[k]
            )

        for j in range(1, half_lateral_points + 1):
            for k in range(len(possible_altitudes)):
                # +j (left)
                bearing = pt[half_forward_points][half_lateral_points + j - 1][
                    k
                ].initialBearingTo(pt[half_forward_points][nb_lateral_points - 1][k])
                total_distance = pt[half_forward_points][half_lateral_points + j - 1][
                    k
                ].distanceTo(pt[half_forward_points][nb_lateral_points - 1][k])
                pt[half_forward_points][half_lateral_points + j][k] = pt[
                    half_forward_points
                ][half_lateral_points + j - 1][k].destination(
                    total_distance / (half_lateral_points - j + 1),
                    bearing,
                    height=possible_altitudes[k],
                )
                # -j (right)
                bearing = pt[half_forward_points][half_lateral_points - j + 1][
                    k
                ].initialBearingTo(pt[half_forward_points][0][k])
                total_distance = pt[half_forward_points][half_lateral_points - j + 1][
                    k
                ].distanceTo(pt[half_forward_points][0][k])
                pt[half_forward_points][half_lateral_points - j][k] = pt[
                    half_forward_points
                ][half_lateral_points - j + 1][k].destination(
                    total_distance / (half_lateral_points - j + 1),
                    bearing,
                    height=possible_altitudes[k],
                )
                for i in range(1, i_initial):
                    alt = pt[i][half_lateral_points][k].height
                    bearing = pt[i - 1][half_lateral_points + j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points + j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    pt[i][half_lateral_points + j][k] = pt[i - 1][
                        half_lateral_points + j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

                    bearing = pt[i - 1][half_lateral_points - j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points - j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    pt[i][half_lateral_points - j][k] = pt[i - 1][
                        half_lateral_points - j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )
                for i in range(i_initial, half_forward_points):
                    # first halp (p0 to np2)
                    bearing = pt[i - 1][half_lateral_points + j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points + j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    pt[i][half_lateral_points + j][k] = pt[i - 1][
                        half_lateral_points + j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=possible_altitudes[k],
                    )

                    bearing = pt[i - 1][half_lateral_points - j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points - j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    pt[i][half_lateral_points - j][k] = pt[i - 1][
                        half_lateral_points - j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=possible_altitudes[k],
                    )
                for i in range(1, half_forward_points - i_final):
                    # second half (np2 to p1)
                    bearing = pt[half_forward_points + i - 1][half_lateral_points + j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points + j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points + j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points + j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=possible_altitudes[k],
                    )

                    bearing = pt[half_forward_points + i - 1][half_lateral_points - j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points - j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points - j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points - j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=possible_altitudes[k],
                    )
                for i in range(half_forward_points - i_final, half_forward_points):
                    alt = pt[half_forward_points + i - 1][half_lateral_points][k].height
                    bearing = pt[half_forward_points + i - 1][half_lateral_points + j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points + j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points + j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points + j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

                    bearing = pt[half_forward_points + i - 1][half_lateral_points - j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points - j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points - j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points - j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

        return pt

    def simple_fuel_loop(self, domain_factory, max_steps: int = 100):

        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=False)
        self.solve_with(solver, domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation: State = self.reset()
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
        fuel = self._get_terminal_state_time_fuel(observation)["fuel"]
        # Evaluate if there is a fuel constraint violation
        enough_fuel = fuel <= self.constraints["fuel"]

        solver._cleanup()
        print(f"fuel : {fuel}, fuel loaded : {self.fuel_loaded}")
        return (fuel, enough_fuel)

    def solve(
        self,
        domain_factory,
        max_steps: int = 100,
        debug: bool = False,
        make_img: bool = False,
    ):
        solver = Astar(heuristic=lambda d, s: d.heuristic(s), debug_logs=debug)
        self.solve_with(solver, domain_factory)
        pause_between_steps = None
        max_steps = 100
        observation = self.reset()

        initial_mass = observation.mass
        initial_time = observation.time

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

            print("step ", i_step)
            print("policy = ", action[0], action[1])
            print("New state = ", observation.pos)
            print("Alt = ", observation.alt)
            print("Mach = ", observation.trajectory.iloc[-1]["mach"])
            print(observation)
            if make_img:
                # update image
                plt.clf()  # clear figure
                clear_output(wait=True)
                figure = self.render(observation)
                # plt.savefig(f'step_{i_step}')

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img:
            if self.res_img_dir:
                plt.savefig(f"{self.res_img_dir}/terminal")
                plt.clf()
                clear_output(wait=True)
                figure = plot_altitude(observation.trajectory)
                plt.savefig(f"{self.res_img_dir}/altitude")
                plot_network(self, dir=self.res_img_dir)
            else:
                plt.savefig(f"terminal")
                plt.clf()
                clear_output(wait=True)
                figure = plot_altitude(observation.trajectory)
                plt.savefig("altitude")
                plot_network(self)
        # goal reached?
        is_goal_reached = self.is_goal(observation)

        final_mass = observation.mass
        final_time = observation.time
        print("Fuel burnt : ", initial_mass - final_mass)
        print("Flight time ", final_time - initial_time)
        terminal_state_constraints = self._get_terminal_state_time_fuel(observation)
        print(terminal_state_constraints)
        if is_goal_reached:
            if self.constraints is not None:
                if self.constraints["time"] is not None:
                    if (
                        self.constraints["time"][1]
                        >= terminal_state_constraints["time"]
                    ):
                        if (
                            self.constraints["fuel"]
                            >= terminal_state_constraints["fuel"]
                        ):
                            print(f"Goal reached after {i_step} steps!")
                        else:
                            print(
                                f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                            )
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but not in the good timelapse!"
                        )
                else:
                    if self.constraints["fuel"] >= terminal_state_constraints["fuel"]:
                        print(f"Goal reached after {i_step} steps!")
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                        )
        else:
            print(f"Goal not reached after {i_step} steps!")
        solver._cleanup()
        return terminal_state_constraints

    def flying(
        self, from_: pd.DataFrame, to_: Tuple[float, float, int]
    ) -> pd.DataFrame:
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
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], to_[0], to_[1])

            def heading(position, destination):
                theta = np.arctan2(
                    np.sin(np.pi / 180.0 * (destination[1] - position[1]))
                    * np.cos(np.pi / 180.0 * destination[0]),
                    np.cos(np.pi / 180.0 * position[0])
                    * np.sin(np.pi / 180.0 * destination[0])
                    - np.sin(np.pi / 180.0 * position[0])
                    * np.cos(np.pi / 180.0 * destination[0])
                    * np.cos(np.pi / 180.0 * (destination[1] - position[1])),
                )
                return theta

            p, _, _ = atmos(alt * ft)
            isobaric = p / 100
            we, wn = 0, 0
            if self.wind_ds:
                time = pos["ts"]
                wind_ms = self.wind_ds.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=alt, t=time
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

            wspd = sqrt(wn * wn + we * we)
            tas = mach2tas(self.mach, alt * ft)  # 400
            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )
            # print("gspeed", gs, "tas ", tas, "bearing ", bearing_degrees)
            if gs * dt > dist:
                # Last step. make sure we go to destination.
                dt = dist / gs
                ll = to_[0], to_[1]
            else:
                ll = latlon(
                    pos["lat"], pos["lon"], d=gs * dt, brg=bearing_degrees, h=alt * ft
                )

            pos["fuel"] = dt * self.fuel_flow(
                pos["mass"],
                tas / kts,
                alt * ft,
                path_angle=(alt - pos["alt"]) / (gs * dt),
            )
            mass = pos["mass"] - pos["fuel"]

            new_row = {
                "ts": pos["ts"] + dt,
                "lat": ll[0],
                "lon": ll[1],
                "mass": mass,
                "mach": self.mach,
                "fuel": pos["fuel"],
                "alt": alt,  # to be modified
            }

            # New distance to the next 'checkpoint'
            dist = distance(
                new_row["lat"], new_row["lon"], to_[0], to_[1], new_row["alt"]
            )

            # print("Dist : %f Dist_ : %f " %(dist,dist_))
            if dist < dist_:
                # print("Fuel new_row : %f" %new_row["fuel"])
                data.append(new_row)
                dist_ = dist
                pos = data[-1]
            else:
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            loop += 1

        return pd.DataFrame(data)

    def speed_management(self, from_: pd.DataFrame) -> float:

        if self.constraints["time"] is None:
            if self.objective == "time" or self.objective == "lazy_time":
                return self.ac["limits"][
                    "MMO"
                ]  # If there is no time constraint & the objective is "time", we use maximum speed
            else:
                return self.ac["cruise"][
                    "mach"
                ]  # If the objecitve is not time, we use cruise speed

        pos = from_.to_dict("records")[0]
        distance_to_goal = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"]),
            LatLon(self.lat2, self.lon2, height=self.alt2),
        )
        new_mach = pos["mach"]
        time = pos["ts"]

        we, wn = 0, 0
        bearing = aero_bearing(pos["lat"], pos["lon"], self.lat2, self.lon2)

        if self.wind_ds:
            wind_ms = self.wind_ds.interpol_wind_classic(
                lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=time
            )

            we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

        wdir = (degrees(atan2(we, wn)) + 180) % 360
        wspd = sqrt(wn * wn + we * we)

        for i in range(10):

            tas = mach2tas(new_mach, pos["alt"] * ft)

            wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

            heading = (360 + bearing - degrees(wca)) % 360

            gsn = tas * cos(radians(heading)) - wn
            gse = tas * sin(radians(heading)) - we

            gs = sqrt(gsn * gsn + gse * gse)

            cost = (distance_to_goal / gs) + time

            if self.constraints["time"][1] is None:

                if (
                    cost >= self.constraints["time"][0]
                    or new_mach >= aircraft(ac)["limits"]["MMO"]
                    or new_mach <= (aircraft(ac)["cruise"]["mach"] - 0.05)
                ):
                    return new_mach
                if cost <= self.constraints["time"][0]:
                    new_mach -= 0.01

            else:
                if (
                    (
                        cost >= self.constraints["time"][0]
                        and cost <= self.constraints["time"][1]
                    )
                    or new_mach >= aircraft(ac)["limits"]["MMO"]
                    or new_mach <= (aircraft(ac)["cruise"]["mach"] - 0.05)
                ):
                    return new_mach

                if cost <= self.constraints["time"][0]:
                    new_mach -= 0.01
                if cost >= self.constraints["time"][1]:
                    new_mach += 0.01

        return new_mach


def compute_gspeed(
    tas: float, true_course: float, wind_speed: float, wind_direction: float
):
    # Tas : speed in m/s
    # course : current bearing
    # wind speed, wind norm in m/s
    # wind_direction : (3pi/2-arctan(north_component/east_component)) in radian
    ws = wind_speed
    wd = wind_direction
    tc = true_course

    # calculate wind correction angle wca and ground speed gs
    swc = ws / tas * sin(wd - tc)
    if abs(swc) >= 1.0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        wca = asin(swc)  # * 180.0 / pi)
        gs = tas * sqrt(1 - swc * swc) - ws * cos(wd - tc)

    if gs < 0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        # Reset possible status message
        error = ""
    return gs


def fuel_optimisation(
    origin: Union[str, tuple],
    destination: Union[str, tuple],
    ac: str,
    constraints: dict,
    wind_interpolator: GenericWindInterpolator,
    mach: float,
) -> float:
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
    while not small_diff:
        fuel_domain_factory = lambda: FlightPlanningDomain(
            origin,
            destination,
            ac,
            constraints=constraints,
            wind_interpolator=wind_interpolator,
            objective="distance",
            nb_forward_points=41,
            nb_lateral_points=11,
            nb_vertical_points=5,
            fuel_loaded=new_fuel,
            mach=mach,
        )
        fuel_domain = fuel_domain_factory()

        fuel_prec = new_fuel
        new_fuel, fuel_remaining = fuel_domain.simple_fuel_loop(fuel_domain_factory)
        step += 1
        small_diff = (fuel_prec - new_fuel) <= (1 / 1000)

    return new_fuel


if __name__ == "__main__":
    """
    Example of launch : python domain.py -o LFPG -d WSSS -ac A388 -obj fuel -w 2023-01-13
    """

    # Definition of command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("-deb", "--debug", action="store_true")
    parser.add_argument(
        "-d", "--destination", help="ICAO code of the destination", type=str
    )
    parser.add_argument("-o", "--origin", help="ICAO code of the origin", type=str)
    parser.add_argument("-ac", "--aircraft", help="ICAO code of the aircraft", type=str)
    parser.add_argument(
        "-obj",
        "--objective",
        help="Objective for the flight (time, fuel, distance)",
        type=str,
    )
    parser.add_argument(
        "-tcs",
        "--timeConstraintStart",
        help="Start Time constraint for the flight. The flight should arrive after that time",
    )
    parser.add_argument(
        "-tce",
        "--timeConstraintEnd",
        help="End Time constraint for the flight. The flight should arrive before that time",
    )
    parser.add_argument(
        "-w",
        "--weather",
        help="Weather day for the weather interpolator, format:YYYYMMDD",
        type=str,
    )
    parser.add_argument(
        "-div",
        "--diversion",
        help="Boolean to put a diversion on the flight plan",
        action="store_true",
    )
    parser.add_argument(
        "-fl",
        "--fuelLoop",
        help="If this option is selected, there will be a first loop to optimise the fuel loaded in the plane",
        action="store_true",
    )
    parser.add_argument(
        "-img",
        "--images",
        help="Saving images from the flight plan e.g. Network points, terminal trajectory...",
        action="store_true",
    )
    parser.add_argument(
        "-cl",
        "--climbSlope",
        help="Climb slope of the aircraft (percent), between 10.0 and 25.0",
        type=float,
    )
    parser.add_argument(
        "-des",
        "--descentSlope",
        help="Descent slope of the aircraft (percent), between 10.0 and 25.0",
        type=float,
    )
    parser.add_argument(
        "-gw",
        "--graphWidth",
        help="Width of the airways graph, take one of these : tiny, small, normal, large, xlarge",
        type=str,
    )

    args = parser.parse_args()

    # Retrieve arguments
    if args.graphWidth:
        if ["tiny", "small", "normal", "large", "xlarge"].__contains__(args.graphWidth):
            graphWidth = args.graphWidth
        else:
            graphWidth = None
    else:
        graphWidth = None

    if args.climbSlope:
        climb_slope = args.climbSlope
    else:
        climb_slope = None
    if args.descentSlope:
        descent_slope = args.descentSlope
    else:
        descent_slope = None

    if args.destination:
        destination = args.destination
    else:
        destination = "LFBO"  # Toulouse
        destination = "EDDB"  # Berlin
        destination = "WSSS"  # Singapour

    if args.origin:
        origin = args.origin
    else:
        origin = "LFPG"

    debug = args.debug

    if args.aircraft:
        ac = args.aircraft
    else:
        ac = "A388"

    if args.objective:
        objective = args.objective
    else:
        objective = None

    if args.timeConstraintStart:
        if args.timeConstraintEnd:
            timeConstraint = (
                float(args.timeConstraintStart),
                float(args.timeConstraintEnd),
            )
        else:
            timeConstraint = (float(args.timeConstraintStart), None)
    else:
        if args.timeConstraintEnd:
            timeConstraint = (0.0, float(args.timeConstraintEnd))
        else:
            timeConstraint = None

    if args.weather:
        if len(args.weather) != 8:
            weather = {
                "year": "2023",
                "month": "01",
                "day": "13",
                "forecast": "nowcast",
            }
        else:
            year = args.weather[0:4]
            month = args.weather[4:6]
            day = args.weather[6:8]
            weather = {"year": year, "month": month, "day": day, "forecast": "nowcast"}
    else:
        weather = None

    diversion = args.diversion
    fuel_loop = args.fuelLoop
    make_img = args.images

    # Define basic constraints

    maxFuel = aircraft(ac)["limits"]["MFC"]
    constraints = {
        "time": timeConstraint,  # Aircraft should arrive before a given time (or in a given window)
        "fuel": maxFuel,
    }  # Aircraft should arrive with some fuel remaining

    # Creating wind interpolator
    if weather:
        wind_interpolator = None
        mat = get_weather_matrix(
            year=weather["year"],
            month=weather["month"],
            day=weather["day"],
            forecast=weather["forecast"],
            delete_npz_from_local=False,
            delete_grib_from_local=False,
        )
        wind_interpolator = GenericWindInterpolator(file_npz=mat)
    else:
        wind_interpolator = None

    # Creating the domain
    domain_factory = lambda: FlightPlanningDomain(
        origin,
        destination,
        ac,
        constraints=constraints,
        wind_interpolator=wind_interpolator,
        objective=objective,
        nb_forward_points=8,
        nb_lateral_points=11,
        fuel_loop=fuel_loop,
        descending_slope=descent_slope,
        climbing_slope=climb_slope,
        graph_width=graphWidth,
    )
    domain = domain_factory()
    # plot_network(domain)
    solvers = match_solvers(domain=domain)
    print(solvers)
    domain.solve(domain_factory, make_img=True, debug=False)

    # athenes paris
    # Fuel burnt :  5374.819150982934
    # Flight time  8397.686199301223
    # {'time': 8397.686199301223, 'fuel': 5374.8191509829085}

    # Fuel
    # burnt: 5715.311946230635
    # Flight
    # time
    # 8940.528209196937
    # {'time': 8940.528209196937, 'fuel': 5715.311946230631}
