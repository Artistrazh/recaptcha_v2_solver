import random
import time
import requests

from loguru import logger


def get_certified_socks() -> str:
    '''Choosing and checking socks IP adress.'''
    socks = None
    while True:
        try:
            socks_current = _get_socks()
            if not socks_current:
                socks = "false"
                # raise EmptySocksError("ERROR::Unable to get socks...")
            socks = socks_current.replace('"', "")
        except (requests.exceptions.RequestException) as socks_err:
            print(f"SOCKS ERROR: {socks_err}")
            time.sleep(1.5)
            continue

        try:
            _test_socks(socks)
        except requests.exceptions.RequestException as req_err:
            print(req_err)
            time.sleep(3)
            continue
        return socks


def _test_socks(socks: str) -> None:
    '''Checking if socks IP is valid.'''
    proxies = dict(http="socks5://" + socks, https="socks5://" + socks)
    requests.get("https://ifconfig.me", proxies=proxies)


def _get_socks() -> str:
    '''Getting random choicing socks.'''
    socks = [
        "0.0.0.0:1080",
        "0.0.0.0:1081",
        "0.0.0.0:1082",
        "0.0.0.0:1083",
        "0.0.0.0:1084"
    ]
    
    proxy = random.choice(socks)

    logger.info("GET SOCKS:", proxy)
    return proxy
