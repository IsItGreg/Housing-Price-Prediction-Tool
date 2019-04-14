import pandas as pd
import numpy as np
from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def main():
    data = pd.read_csv("ass.csv")
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    newarr = pd.DataFrame({})
    newarr["LAND_SF"] = data.pop("LAND_SF")
    newarr["YR_BUILT"] = data.pop("YR_BUILT")
    newarr["NUM_FLOORS"] = data.pop("NUM_FLOORS")
    newarr["NUM_PARK"] = data.pop("U_NUM_PARK")
    newarr["NUM_RMS"] = data.pop("T_TOT_RMS")
    newarr["NUM_BDRMS"] = data.pop("T_BDRMS")
    newarr["NUM_FULL_BTH"] = data.pop("T_FULL_BTH")
    newarr["NUM_HALF_BTH"] = data.pop("T_HALF_BTH")

    latlong = pd.DataFrame({'LAT': [], 'LON': []})

    for i, row in data.iterrows():
        addr = row['ST_NUM'] + ' ' + row['ST_NAME'] + ' ' + row['ST_NAME_SUF'] + ' Boston, MA'
        location = geolocator.geocode(addr)
        if location is None:
            latlong.loc[i] = [42.3602534, -71.0582912]
        else:
            latlong.loc[i] = [location.latitude, location.longitude]

    newarr["LAT"] = latlong.pop("LAT")
    newarr["LON"] = latlong.pop("LON")
    newarr["AV_TOTAL"] = data.pop("AV_TOTAL")

    newarr.to_csv('formatted.csv', index=False)


if __name__ == '__main__':
    main()
