# Add your Python code here.

# Previous code...

    # NOTE: Open-Meteo's...
    az_open = cfg.azimuth_deg_from_north - 180.0
    # wrap to [-180, 180]
    az_open = ((az_open + 180.0) % 360.0) - 180.0

# Following code...