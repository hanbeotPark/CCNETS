'''
    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
        Kim, JinSu, wlstnwkwjsrj@naver.com
'''
import json
from argparse import ArgumentParser

def write_parser(args):
    with open("../ccnets/config.txt","w") as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

def get_parser():
    parser = ArgumentParser()
    args = parser.parse_args(args = [])
    with open("../ccnets/config.txt", 'r') as f:
        args.__dict__ = json.load(f)
    f.close()
    return args
