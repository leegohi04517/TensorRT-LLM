import time

class Timer(object):
    def __init__(self, name=None, logger=None):
        self.name = name if name else ''
        if not logger:
            import logging
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        now = time.time()
        self.logger.info('%s cost:%s' % (self.name, now - self.start_time))

if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(20)
    logging.info('test')
    with Timer('aaa'):
        range(1000)
