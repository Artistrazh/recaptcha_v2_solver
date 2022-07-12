from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.remote.webdriver import WebDriver, WebElement


def get_chromedriver(socks: str, user_agent: str) -> WebDriver: 
    '''Getting Chrome Driver object with using socks and User Agent.'''
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument(f"--proxy-server=socks5://{socks}")
    chrome_options.add_argument("--user-agent=%s" % user_agent)
    chrome_options.add_experimental_option("prefs", {"intl.accept_languages": "en-US,en"})

    # chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument('--headless')
    chrome_options.add_experimental_option("useAutomationExtension", False)

    chrome_options.add_argument('--profile-directory=Default')
    chrome_options.add_argument("--disable-plugins-discovery")

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "ignore-certificate-errors",
                                                               "safebrowsing-disable-download-protection",
                                                               "safebrowsing-disable-auto-update",
                                                               "disable-client-side-phishing-detection"])
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    caps = DesiredCapabilities.CHROME

    now = datetime.now()
    formated_time = now.strftime("%H:%M:%S %d.%m.%Y")

    capabilities = {
        "name": f"Recaptcha v2 Solver {formated_time}",
        "browserName": "chrome",
        "browserVersion": "latest",
        "selenoid:options": {
            "enableVNC": True,
            "enableVideo": False
        }
    }

    capabilities.update(chrome_options.to_capabilities())
    capabilities.update(caps)
    
    driver = webdriver.Remote(command_executor="http://0.0.0.0:4444/wd/hub", desired_capabilities=capabilities)
    return driver
