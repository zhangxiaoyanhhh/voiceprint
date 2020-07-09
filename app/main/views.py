import os
import time
import json
import wave as wave
from flask import render_template, redirect, flash, request, send_file, jsonify
from sqlalchemy.dialects.mysql import pymysql
from app.main.model import Session, Tvoice,engine
from app import get_logger, get_config
from . import main
from app.main.feature import load_train,load_test
from app.main.ubm import train, predict, pred_ubm
from .LSTM import test, train2, pred_lstm
from .svm import svm_two, pred_svm1
from sklearn.externals import joblib
import numpy as np
from sklearn import svm
import time
from .svm2 import svm_2, pred_svm2
from .svm_1 import svm_one

cfg = get_config()
logger = get_logger(__file__)
path = os.path.abspath(os.path.dirname(__file__)).replace("main", "")
session = Session()


@main.route("/index")
def index():
    return render_template("index.html")


@main.route("/api/shengwen", methods=["POST"])
def raw_file_upload():
    code = "succ"
    try:
        raw_file = request.files.get("audio")
        name = request.form.get("name")
        raw_file.save(path + "/voice/train/" + name + ".wav")
        path_train = path + "/voice/train/" + name + ".wav"
        data, name1 = load_train(name, path_train)
        print(name1)
    except:
        code= "err"
    return jsonify({"data":{"code":code}})


@main.route("/api/svm1", methods=["POST"])
def test_shengwen1():
    # print("svm1")

    raw_file = request.files.get("audio")
    raw_file.save(path + "/voice/test/test.wav")
    code = "succ"
    name = " "
    # print(code)
    # SVM1
    try:
        start = time.time()
        score,name_label = pred_svm1()
        print(score,name_label)
        b = session.query(Tvoice).all()
        train_y = []
        for i in b:
            train_y.append(i.label)
        if score>500:
            end = time.time()
            name = "SVM1测试结果："+train_y[name_label] +": " +str(score/800) + "  测试时间{:.5}s".format(end-start)
        else:
            name = "用户未注册"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code,"name":name}})


@main.route("/api/svm2", methods=["POST"])
def test_shengwen2():
    raw_file = request.files.get("audio")
    raw_file.save(path + "/voice/test/test.wav")
    code = "succ"
    name = " "
    try:
        start = time.time()
        score, name_label = pred_svm2()
        print(score, name_label)
        b = session.query(Tvoice).all()
        train_y = []
        for i in b:
            train_y.append(i.label)
        if score > 500:
            end = time.time()
            name = "SVM2测试结果："+train_y[name_label] +": " +str(score/800) + "  测试时间{:.5}s".format(end-start)
        else:
            name = "用户未注册"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code,"name":name}})


@main.route("/api/ubm", methods=["POST"])
def test_shengwen3():
    raw_file = request.files.get("audio")
    raw_file.save(path + "/voice/test/test.wav")
    code = "succ"
    name = " "
    try:
        start = time.time()
        TOTAL = 0
        train_y = []
        b = session.query(Tvoice).all()
        for i in b:
            train_y.append(i.label)
            TOTAL += 1
        score, name_label = pred_ubm(TOTAL)
        print(score, name_label)

        if score > 500:
            end = time.time()
            name = "GMM-UBM测试结果："+train_y[name_label] +": " +str(score/800) + "  测试时间{:.5}s".format(end-start)
        else:
            name = "用户未注册"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code,"name":name}})


@main.route("/api/lstm", methods=["POST"])
def test_shengwen4():
    raw_file = request.files.get("audio")
    raw_file.save(path + "/voice/test/test.wav")
    code = "succ"
    name = " "
    try:
        start = time.time()
        train_y = []
        b = session.query(Tvoice).all()
        for i in b:
            train_y.append(i.label)
        score, name_label = pred_lstm()
        print(score, name_label)

        if score > 500:
            end = time.time()
            name = "LSTM测试结果："+train_y[name_label] +": " +str(score/800) + "  测试时间{:.5}s".format(end-start)
        else:
            name = "用户未注册"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code,"name":name}})


@main.route("/api/fus", methods=["POST"])
def test_shengwen5():
    raw_file = request.files.get("audio")
    raw_file.save(path + "/voice/test/test.wav")
    code = "succ"
    name = " "
    try:
        start = time.time()
        TOTAL = 0
        train_y = []
        b = session.query(Tvoice).all()
        for i in b:
            train_y.append(i.label)
            TOTAL += 1
        score1, name_label1 = pred_ubm(TOTAL)
        score2, name_label2 = pred_lstm()
        score3, name_label3 = pred_svm2()
        score4, name_label4 = pred_svm1()
        # print(score1,name_label1)
        # print(score2, name_label2)
        # print(score3, name_label3)
        # print(score4, name_label4)
        score = np.hstack((score1, score2, score3, score4))
        name_label = np.hstack((name_label1, name_label2, name_label3, name_label4))
        print(score,name_label)
        label = np.argmax(np.bincount(name_label))
        x = np.argwhere(label == name_label)
        # print(x)
        if len(x)>=3:
            s = []
            for x1 in x:
                s.append(score[x1[0]])
            # print(s)
            max_score = max(s)
            if max_score > 500:
                end = time.time()
                name ="模型融合测试结果："+train_y[label] +": " +str(max_score/800) + "  测试时间{:.5}s".format(end-start)
            else:
                name = "用户未注册"
        else:
            name = "用户未注册"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code,"name":name}})


@main.route("/api/test2", methods=["POST"])
def test_shengwen0():
    raw_file = request.files.get("audio")
    name = request.form.get("name")
    # print(type(name))
    raw_file.save(path + "/voice/test/test.wav")
    try:
        TOTAL = 0
        train_y = []
        b = session.query(Tvoice).all()
        for i in b:
            train_y.append(i.label)
            TOTAL += 1
        score1, name_label1 = pred_ubm(TOTAL)
        score2, name_label2 = pred_lstm()
        score3, name_label3 = pred_svm2()
        score4, name_label4 = pred_svm1()
        # print(score1,name_label1)
        # print(score2, name_label2)
        # print(score3, name_label3)
        # print(score4, name_label4)
        score = np.hstack((score1, score2, score3, score4))
        name_label = np.hstack((name_label1, name_label2, name_label3, name_label4))
        print(score, name_label)
        label = np.argmax(np.bincount(name_label))
        x = np.argwhere(label == name_label)
        # print(x)
        if len(x) >= 3:
            s = []
            for x1 in x:
                s.append(score[x1[0]])
            # print(s)
            max_score = max(s)
            if name == train_y[label] and max_score > 500:
                code = "succ"
            else:
                code = "err"
        else:
            code = "err"
    except Exception as e:
        print(e)
        code = "err"
    return jsonify({"data":{"code":code}})


@main.route("/api/train1", methods=["POST"])
def train_shengwen1():
    code = "succ"
    try:
        svm_one()
    except:
        code = "err"

    return jsonify({"data":{"code":code}})


@main.route("/api/train2", methods=["POST"])
def train_shengwen2():
    code = "succ"
    try:
        svm_2()
    except Exception as e:
        print(e)
        code = "err"

    return jsonify({"data":{"code":code}})


@main.route("/api/train3", methods=["POST"])
def train_shengwen3():
    code = "succ"
    try:
        b = session.query(Tvoice).all()
        out = 0
        for i in b:
            out += 1
        train(out)
    except:
        code = "err"

    return jsonify({"data":{"code":code}})

@main.route("/api/train4", methods=["POST"])
def train_shengwen4():
    code = "succ"
    try:
        train2()
    except:
        code = "err"

    return jsonify({"data":{"code":code}})
