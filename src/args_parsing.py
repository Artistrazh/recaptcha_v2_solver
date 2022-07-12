import argparse

def server_parser() -> argparse.ArgumentParser:
    '''Parsing server part argument.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    return parser

def solver_parser() -> argparse.ArgumentParser:
    '''Parsing RecaptchaV2 solver arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--links', default='https://2captcha.com/ru/demo/recaptcha-v2', type=str)
    parser.add_argument('--socks', type=str)
    return parser
