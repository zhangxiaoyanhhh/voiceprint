from flask import render_template,redirect,url_for
from . import main
from app import get_logger

logger = get_logger(__name__)

@main.app_errorhandler(404)
def page_not_found(e):
    #return render_template('errors/404.html'), 404
    logger.exception(e)
    return redirect(url_for('main.index'))


@main.app_errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return render_template('errors/500.html'), 500
