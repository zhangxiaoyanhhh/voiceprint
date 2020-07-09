# coding=utf-8
from __future__ import unicode_literals, absolute_import
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pymysql

pymysql.install_as_MySQLdb()

engine = create_engine("mysql+mysqldb://root:zxy153925@localhost:3306/audio")

Session = sessionmaker(bind=engine)

ModelBase = declarative_base() #<-元类


class Tvoice(ModelBase):
    __tablename__ = "audio1"

    id = Column(Integer, primary_key=True)
    label = Column(String(255), nullable=False, comment="姓名")
    feature = Column(LargeBinary(length=16777216), comment="特征")
    qian = Column(Integer, comment="长")
    hou = Column(Integer, comment="宽")

class Tvoice1(ModelBase):
    __tablename__ = "audio2"

    id = Column(Integer, primary_key=True)
    label = Column(String(255), nullable=False, comment="姓名")
    feature = Column(LargeBinary(length=16777216), comment="特征")
    one = Column(Integer, comment="长")
    two = Column(Integer, comment="宽")
    three = Column(Integer, comment="高")