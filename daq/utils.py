from datetime import date, datetime

PRESTO_ADDRESS = "172.23.20.29"
PRESTO_PORT = None  # Use system default port

def get_date_str():
    """Get the current date as a string in the format YYYYMMDD
    """
    today = date.today()
    date_str = f"{today.year}{today.month:02d}{today.day:02d}"
    return date_str

def get_date_str_with_time():
    """Get the current date and time as a string in the format YYYYMMDD_HHMMSS
    """
    today = datetime.now()
    date_str = f"{today.year}{today.month:02d}{today.day:02d}_{today.hour:02d}{today.minute:02d}{today.second:02d}"
    return date_str

def get_presto_address():
    """Get the address of the presto device.
    """
    return PRESTO_ADDRESS


def get_presto_port():
    """Get the port of the presto device.
    """
    return PRESTO_PORT   