import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config():
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret'
    ITEMS_PER_PAGE = 10
    DEBUG = False

    @staticmethod
    def init_app(app):
        pass




class TestingConfig(Config):
    DEBUG = True



config = {
    'default': TestingConfig
}
