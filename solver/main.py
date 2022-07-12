import os
import sys
import time
import json
import random
import asyncio
import requests
import warnings
from typing import Dict, Union

from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from proxy import get_certified_socks
from const import user_agent, test_links
from chromedriver import get_chromedriver

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    import src.args_parsing
    parser = src.args_parsing.solver_parser()
    args = parser.parse_args()

def check_element(driver, xpath: str) -> bool:
    '''Check element on page by Xpath.'''
    time.sleep(2)
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


def iframe_with_picture(driver) -> bool:
    '''Switch to iframe with image if it exists otherwise false.'''
    #Switch to a new iframe with an image
    if check_element(driver, "//iframe[@title='recaptcha challenge expires in two minutes']"):
        driver.switch_to.frame(driver.find_elements_by_xpath("//iframe[@title='recaptcha challenge expires in two minutes']")[0])
        return True
    return False


def check_solved(driver) -> bool:
    '''Check if the captcha is solved.'''
    # Switch to main HTML document
    driver.switch_to.default_content()

    # Switch to iframe reCAPTCHA
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//iframe[@title='reCAPTCHA']")))
    driver.switch_to.frame(driver.find_elements_by_xpath("//iframe[@title='reCAPTCHA']")[0])

    # Check if captcha is solved or not
    if check_element(driver, "//div[@class='recaptcha-checkbox-border' and @style='display: none;']"):
        logger.info("Captcha solved!")
        return True
    return False


# TODO Make a check for the red inscription (in case of unsuccessful passing of the captcha)
def check_red_inscription(driver) -> bool:
    '''Processing of red inscriptions.'''
    if check_element(driver, "//div[contains(@class, 'rc-imageselect-error')][@tabindex='0']"):
        logger.info("Red inscription!")
        return True
    elif check_element(driver, "//div[contains(@class, 'rc-imageselect-incorrect')][@tabindex='0']"):
        logger.info("Red inscription!")
        return True
    return False
  

def send_big_picture(text: str, link_to_picture: str, squares: int) -> list:
    '''Sending a big picture to the endpoint.'''
    info = {
        "text": text,
        "link_to_picture": link_to_picture,
        "squares": squares
    }

    data = requests.post("http://0.0.0.0:11013/send_big_picture", json=info)
    print(data.text)
    solvers = json.loads(data.text)
    return solvers


def send_small_picture(text: str, pictures: str) -> list:
    '''Sending a small picture to the endpoint.'''
    info = {
        "text": text,
        "pictures": pictures, # {"3": links, "5": links}
        "squares": 1
    }

    data = requests.post("http://0.0.0.0:11013/send_small_picture", json=info)
    print(data.text)
    solvers = json.loads(data.text)
    return solvers


def main_pipeline(driver) -> Union[str, str, int]:
    '''Send big picture.'''
    # Get text
    wait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//div[contains(@class, 'rc-imageselect-desc')]/strong")))
    text = driver.find_elements_by_xpath("//div[contains(@class, 'rc-imageselect-desc')]/strong")[0].text
    print("text:", text)

    # Get link picture
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='rc-image-tile-wrapper']/img")))
    img = driver.find_elements_by_xpath("//div[@class='rc-image-tile-wrapper']/img")[0]
    link_to_picture = img.get_attribute("src")
    print("link_to_picture:", link_to_picture)

    # Get squares
    squares = 9 if check_element(driver, "//table[@class='rc-imageselect-table-33']") else 16
    print("squares:", squares)
    
    return text, link_to_picture, squares


def second_pipeline(driver, previous_solvers: int) -> Union[str, str]:
    '''Sending a small picture.'''
    # Get text
    wait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//div[contains(@class, 'rc-imageselect-desc')]/strong")))
    text = driver.find_elements_by_xpath("//div[contains(@class, 'rc-imageselect-desc')]/strong")[0].text
    print("text:", text)

    # Get links pictures
    pictures = {} # {"3": links, "5": links}
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='rc-image-tile-wrapper']/img")))
    for solver in previous_solvers:
        solver = int(solver)
        img = driver.find_elements_by_xpath("//div[@class='rc-image-tile-wrapper']/img")[solver-1]
        pictures[str(solver)] = img.get_attribute("src")
    print("pictures:", pictures)

    return text, pictures


def clicks(driver, solvers: int):
    '''Clicks on pictures.'''
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//td[@role='button'][@class='rc-imageselect-tile'][@tabindex]")))
    buttons = driver.find_elements_by_xpath("//td[@role='button'][@class='rc-imageselect-tile'][@tabindex]")
    for solver in solvers:
        solver = int(solver)
        driver.execute_script("arguments[0].scrollIntoView();", buttons[solver-1])
        buttons[solver-1].click()
        time.sleep(random.uniform(0.6, 1.5))


def verify_button(driver, solvers: int):
    '''Confirm Recaptcha.'''
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//button[@id='recaptcha-verify-button']")))

    time.sleep(random.uniform(0.6, 1.5))
    verify_button = driver.find_elements_by_xpath("//button[@id='recaptcha-verify-button']")[0]
    driver.execute_script("arguments[0].scrollIntoView();", verify_button)
    time.sleep(random.uniform(0.6, 1.5))

    if verify_button.text == "VERIFY":
        if solvers:
            verify_button.click()
            print(verify_button.text, "click!")
        else:
            print(verify_button.text, "not click!")
    elif verify_button.text == "NEXT":
        verify_button.click()
        print(verify_button.text, "click!")
    else: # SKIP
        verify_button.click()
        print(verify_button.text, "click!")


def version_1(driver, v2=None) -> Union[bool, int]:
    '''Implementation of the big picture sending logic.'''
    
    # Option 1: one big picture
    # just press the buttons and press confirmation
    
    # Get main parameters
    text, link_to_picture, squares = main_pipeline(driver)

    # Sending data and getting a response
    solvers = send_big_picture(text, link_to_picture, squares)

    if solvers:
        # Click on selected pictures
        clicks(driver, solvers)

    # Confirm Recaptcha (verify, next, skip)
    if v2 is None:
        verify_button(driver, solvers)

    if solvers:
        # Checking the red lettering
        status_inscription = check_red_inscription(driver)
        if status_inscription:
            return "red", solvers

        # Checking Recaptcha
        if check_solved(driver):
            return True, solvers

    return False, solvers


def version_2(driver, previous_solvers):
    '''Implementation of the logic for sending a small picture.'''
    
    # Option 2: one large picture and many small ones appear later
    # In this case, small pictures are loaded and they must be served, 
    # saved separately, sent to the verification service and clicked if they match.
    # 

    # Get second parameters
    text, pictures = second_pipeline(driver, previous_solvers)

    # Sending data and getting a response
    if pictures:
        solvers = send_small_picture(text, pictures)

        if solvers:
            # Click on selected pictures
            clicks(driver, solvers)
    else:
        solvers = []

    # Confirm Recaptcha (verify, next, skip)
    if not solvers:
        verify_button(driver, True)

        # Checking the red lettering
        status_inscription = check_red_inscription(driver)
        if status_inscription:
            return "red", solvers

        # Checking Recaptcha
        if check_solved(driver):
            return True, solvers

    return False, solvers


async def solver(driver, RECAPTCHA_LINK: str) -> Union[str, bool]:
    '''Determines the type of captcha on the fly and, depending on this, calls different pipelines.'''
    driver.get(RECAPTCHA_LINK)

    # Switch to iframe reCAPTCHA
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//iframe[@title='reCAPTCHA']")))
    driver.switch_to.frame(driver.find_elements_by_xpath("//iframe[@title='reCAPTCHA']")[0])

    # Find and click on the checkbox
    wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='recaptcha-checkbox-border']")))
    checkbox = driver.find_elements_by_xpath("//div[@class='recaptcha-checkbox-border']")[0]
    driver.execute_script("arguments[0].scrollIntoView();", checkbox)
    checkbox.click()

    # Checking solved captcha on click or not
    if check_element(driver, "//div[@class='recaptcha-checkbox-checkmark'][@style]"):
        print("Captcha solved on click!")
        return "on click", driver

    solvers = []
    v2 = 0
    while True:
        # Switch to main HTML document
        driver.switch_to.default_content()

        if check_element(driver, "//div[@id='recaptcha-accessible-status'][contains(text(), 'Recaptcha requires verification.')]"):
            return False, driver

        # Switch to a new iframe with an image
        if not iframe_with_picture(driver): # If there is no iframe, start pipline from the beginning
            return False, driver

        # We check the Recaptcha option and run 1 or 2
        if check_element(driver, "//span[@class='rc-imageselect-carousel-instructions']"):
            print("Option 1 [Carousel] - many captcha")
            try:
                status, solvers = version_1(driver)
                # print(status, solvers)
                if status == "red":
                    return False, driver
                elif status:
                    return status, driver
            except Exception as ex: # TODO: Make except timeout when the time for passing the Recaptcha is over
                print('error', ex)
                return False, driver
            continue
        elif check_element(driver, "//div[contains(@class, 'rc-imageselect-desc')]/span"):
            print("Option 2 [Disappearing Pictures]")
            try:
                if v2 == 0:
                    status, solvers = version_1(driver, v2)
                    v2 = 1
                else:
                    status, solvers = version_2(driver, solvers)
                # print(status, solvers)
                if status == "red":
                    return False, driver
                elif status:
                    return status, driver
            except Exception as ex: # TODO: 
                print('error', ex)
                return False, driver
            continue
        else:
            print("Option 1 [Carousel] - one captcha")
            try:
                status, solvers = version_1(driver)
                # print(status, solvers)
                if status == "red" or status == False:
                    return False, driver
                elif status:
                    return status, driver
            except Exception as ex:
                print('error', ex)
                return False, driver
            continue


async def main() -> Union[None, str]:
    '''Start attempting to solve captcha.'''
    def choice_socks() -> str:
        if args.socks is None:
            socks = get_certified_socks()
        else:
            socks = args.socks
        print('SOCKS is', socks)
        return socks
    driver = get_chromedriver(choice_socks(), user_agent)

    s = 0 # how many times will the test pass successfully
    c = 0 # how many times will the test be passed without offering a picture
    all = 0
    total_attempts = 1000

    for i in range(total_attempts):
        print("ALL = ", all)
        if (all % 100) == 0 and all != 0: # starting new session every 10 attempts
            print("New session!")
            driver.quit()
            driver = get_chromedriver(choice_socks(), user_agent)

        try:
            if args.links is None:
                RECAPTCHA_LINK = random.choice(test_links)
            else:
                RECAPTCHA_LINK = args.links
            print("RECAPTCHA_LINK:", RECAPTCHA_LINK)
            status, driver = await asyncio.wait_for(solver(driver, RECAPTCHA_LINK), timeout=125.0)
            all += 1
            if status == "on click":
                c+=1
                print(f"on click {c} out of {all}")
            elif status:
                s+=1
                with open('log.txt', 'w') as f:
                    f.write(f"successful {s} out of {all}\n")
                logger.info(f"successful {s} out of {all}")
        except asyncio.TimeoutError:
            logger.error('Timeout!')
        except Exception as ex:
            logger.error('Exception:', ex)


if __name__ == "__main__":
    asyncio.run(main())

    
