import os
from time import sleep
from typing import Any
import xarray as xr
import matplotlib.pyplot as plt
from flightplanning_utils import (
    WindInterpolator,
    flying,
    plot_trajectory, 
)
from openap.prop import aircraft
from openap import aero
from domain import FlightPlanningDomain, Action
from pygeodesy.ellipsoidalVincenty import LatLon
from copy import deepcopy
from time import perf_counter

def debug():
    filename = 'instance3.grib' 
    origin = 'LFPG' 
    destination = 'WSSS' 
    wind_interpolator: WindInterpolator = None
    file = None
    if filename is not None :
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), "instances/%s" % filename))
    if file:
        wind_interpolator = WindInterpolator(file)
    axes = wind_interpolator.plot_wind(alt=35000.0, t=[0], plot_wind=True)
    plt.savefig('wind')
    #if wind_interpolator:
    #    wind_dataset = wind_interpolator.get_dataset()
    #    wind_dataset.u.values -= 60
    #    axes = wind_interpolator.plot_wind(alt=35000.0, t=[0], plot_wind=True)
    #    plt.savefig('wind')
    #    plt.show()
    domain = FlightPlanningDomain(origin, destination, "A388", wind_interpolator=wind_interpolator)

    domain_factory = lambda: domain
    initial_state = domain.get_initial_state()
    available_actions = domain.get_applicable_actions(initial_state).get_elements()
    print(available_actions)
    print(initial_state)

    

    current_state = deepcopy(initial_state)
    trajectory = [current_state]
    network = domain.network

    while not domain.is_terminal(current_state):
        action = Action.straight
        # get corresponding action
        t = perf_counter()
        outcome = domain.get_next_state(current_state, action)
        t_end = perf_counter()
        print(f"{t_end-t} seconds for get next state")
        current_state = outcome
        trajectory += [deepcopy(current_state)]
        id_waypoints = current_state.pos
        point_graph = LatLon(network[id_waypoints[0]][id_waypoints[1]].lat,
                             network[id_waypoints[0]][id_waypoints[1]].lon)
        point_state = LatLon(current_state.trajectory.iloc[-1].lat, 
                             current_state.trajectory.iloc[-1].lon)
        print("Distance", aero.distance(lat1=point_graph.lat, 
                                        lon1=point_graph.lon, 
                                        lat2=point_state.lat, 
                                        lon2=point_state.lon))
        print("Point in the graph : ", 
              network[id_waypoints[0]][id_waypoints[1]].lat, network[id_waypoints[0]][id_waypoints[1]].lon)
        print("Point of the state ", current_state.trajectory.iloc[-1].lat, current_state.trajectory.iloc[-1].lon)
    print("arrived ")
    last_state = trajectory[-1]

    full_traj = last_state.trajectory
    lats = full_traj["lat"]
    longs = full_traj["lon"]

    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs
    from cartopy.feature import BORDERS, LAND, OCEAN

    origin_coord = domain.lat1, domain.lon1
    target_coord = domain.lat2, domain.lon2


    fig, ax = plt.subplots(1, subplot_kw={"projection": ccrs.PlateCarree()}) 
    ax.set_extent([min(origin_coord[1], target_coord[1]) - 4, max(origin_coord[1], target_coord[1])+4, 
                   min(origin_coord[0], target_coord[0]) - 2, max(origin_coord[0], target_coord[0]) + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")
    ax.plot(longs, lats, marker="o", transform=ccrs.Geodetic())
    ax.scatter([network[x][x1].lon for x in range(len(network)) 
                for x1 in range(len(network[x]))], 
                [network[x][x1].lat for x in range(len(network)) 
                for x1 in range(len(network[x]))], transform=ccrs.Geodetic(), s=0.2)
    #ax.stock_img()
    fig.savefig("network points.png")


if __name__ == "__main__":
    debug()