# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from datetime import datetime, timedelta
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from tempfile import NamedTemporaryFile
from typing import Callable, Collection, Iterable, Tuple, Union
from time import process_time
import cdsapi
import cfgrib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.figure import Figure
from openap import aero, nav
from weather_interpolator.weather_tools import get_weather_noaa

# from openap.extra.aero import ft, h_isa
# from openap.top import wind
from scipy.interpolate import RegularGridInterpolator
from weather_interpolator.weather_tools.interpolator.GenericInterpolator import GenericWindInterpolator

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print("Elapsed: %s" % (time.time() - self.tstart))

def flying(
    from_: pd.DataFrame, to_: Tuple[float, float], ds: GenericWindInterpolator, fflow: Callable
) -> pd.DataFrame:
    """Compute the trajectory of a flying object from a given point to a given point

    Args:
        from_ (pd.DataFrame): the trajectory of the object so far
        to_ (Tuple[float, float]): the destination of the object
        ds (xr.Dataset): dataset containing the wind field
        fflow (Callable): fuel flow function

    Returns:
        pd.DataFrame: the final trajectory of the object
    """
    pos = from_.to_dict("records")[0]

    dist_ = aero.distance(pos["lat"], pos["lon"], to_[0], to_[1], pos["alt"])
    data = []
    epsilon = 100
    dt = 600
    dist = dist_
    loop = 0
    while dist > epsilon:  # or loop < 20 or dt > 0:
        bearing = aero.bearing(pos["lat"], pos["lon"], to_[0], to_[1])
        p, _, _ = aero.atmos(pos["alt"] * aero.ft)
        isobaric = p / 100
        we, wn = 0, 0
        if ds:
            time = pos["ts"]
            wind_ms = ds.interpol_wind_classic(
                lat=pos["lat"],
                longi=pos["lon"],
                alt=pos["alt"],
                t=time
            )
            we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

        wdir = (degrees(atan2(we, wn)) + 180) % 360
        wspd = sqrt(wn * wn + we * we)

        tas = aero.mach2tas(pos["mach"], pos["alt"])  # 400

        wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

        # ground_speed = sqrt(
        #     tas * tas
        #     + wspd * wspd
        #     + 2 * tas * wspd * cos(radians(bearing - wdir - wca))
        # )

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
            ll = aero.latlon(pos["lat"], pos["lon"], gs * dt, brg, pos["alt"])
        pos["fuel"] = dt * fflow(pos["mass"], 
                                 tas / aero.kts, 
                                 pos["alt"] * aero.ft, 
                                 path_angle=0.0)
        mass = pos["mass"] - pos["fuel"]

        new_row = {
            "ts": pos["ts"] + dt,
            "lat": ll[0],
            "lon": ll[1],
            "mass": mass,
            "mach": pos["mach"],
            "fuel": pos["fuel"],
            "alt": pos["alt"],
        }

        # New distance to the next 'checkpoint'
        dist = aero.distance(
            new_row["lat"], new_row["lon"], to_[0], to_[1], new_row["alt"]
        )
        
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


def plot_trajectory(
    lat1, lon1, lat2, lon2, trajectory: pd.DataFrame, ds: xr.Dataset
) -> Figure:
    """Plot the trajectory of an object

    Args:
        trajectory (pd.DataFrame): the trajectory of the object
        ds (xr.Dataset): the dataset containing the wind field

    Returns:
        Figure: the figure
    """

    fig = Figure(figsize=(600, 600))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.resizable = False
    fig.set_dpi(1)

    # lon1, lat1 = trajectory.iloc[0]["lon"], trajectory.iloc[0]["lat"]
    # lon2, lat2 = trajectory.iloc[-1]["lon"], trajectory.iloc[-1]["lat"]

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    ax = plt.axes(
        projection=ccrs.TransverseMercator(
            central_longitude=(lonmax - lonmin) / 2,
            central_latitude=(latmax - latmin) / 2,
        )
    )

    wind_sample = 30

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    if ds is not None:
        t = trajectory.ts.iloc[-1]
        alt = trajectory.alt.iloc[-1]
        """
        self.wind_interpolator.plot_wind(
            alt=alt,
            lon_min=max(-180, lonmin - 4),
            lon_max=min(+180, lonmax + 4),
            lat_min=max(-90, latmin - 2),
            lat_max=min(+90, latmax + 2),
            t=int(t),
            n_lat=180,
            n_lon=720,
            plot_wind=False,
            plot_barbs=False,
            ax=ax,
        )
        """

    # great circle
    ax.scatter(lon1, lat1, c="darkgreen", transform=ccrs.Geodetic())
    ax.scatter(lon2, lat2, c="tab:red", transform=ccrs.Geodetic())

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="tab:red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # trajectory
    ax.plot(
        trajectory.lon,
        trajectory.lat,
        color="tab:green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="Optimal",
    )

    ax.legend()

    # Save it to a temporary buffer.
    # buf = BytesIO()
    # fig.savefig(buf, format="png")
    # Embed the result in the html output.
    # data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return fig


if __name__ == "__main__":
    get_weather_noaa.load_npz()
