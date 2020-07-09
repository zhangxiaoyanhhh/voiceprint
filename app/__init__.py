# encoding=utf8
from flask import Flask, request, g
from conf.config import config
import logging
from logging.config import fileConfig
import os
import time
import json
from flask_cors import *


# 加载配置文件
from os import path
log_file_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'conf/log-app.conf')
# print(log_file_path)
fileConfig(log_file_path)
# fileConfig('conf/log-app.conf')


def get_logger(name):
    return logging.getLogger(name)


# 反sql注入
def anti_inj(param):
    inj_str = "'|and|exec|insert|select|delete|update|count|*|%|chr|mid|master|truncate|char|declare|;|or|-|+|(|)"
    for x in inj_str.strip().split('|'):
        param = param.replace(x, '')
    return param


def get_basedir():
    return os.path.abspath(os.path.dirname(__file__))


def get_config():
    return config[os.getenv('FLASK_CONFIG') or 'default']


# 为计算响应延迟
def __before_request():
    g.requestId = str(int(time.time() * 1000000))
    get_logger(__name__).debug("Start Once Access, and this requestId is %s" % g.requestId)


# 为记录收发日志
def __after_request(response):
    logging.debug(json.dumps({
        "AccessLog": {
            "status_code": response.status_code,
            "method": request.method,
            "form": request.values.to_dict() if type(request.values) != dict else request.values,
            "json": request.get_json(force=True, silent=True),
            "ip": request.headers.get('X-Real-Ip', request.remote_addr),
            "url": request.url,
            "referer": request.headers.get('Referer'),
            "agent": request.headers.get("User-Agent"),
            "requestId": str(g.requestId),
        }
    }, ensure_ascii=False
    )[0:5000])
    # print(response.get_data().decode())
    return response


cfg = get_config()


def create_app(config_name, agent=False):
    app = Flask(__name__,template_folder="templates",static_folder="static")
    CORS(app, supports_credentials=True)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    app.before_request(__before_request)
    app.after_request(__after_request)
    return app
