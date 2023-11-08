import configparser
import logging

_boolean_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                   '0': False, 'no': False, 'false': False, 'off': False}

def getboolean(v):
    if v.lower() not in _boolean_states:
        raise(ValueError, 'Not a boolean: %s' % v)
    return _boolean_states[v.lower()]


class Valuer(object):
    def __init__(self, dict=None):
        if dict is None:
            self._dict = {}
        else:
            self._dict = dict

    def __getattr__(self, name):
        try:
            return super(Valuer, self).__getattr__(name)
        except:
            return self._dict.get(name)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        self._dict[name] = value

    def items():
        return self._dict.keys()

    def __repr__(self):
        return str(self._dict)


class Conf(Valuer):
    def __init__(self, path):
        super(Conf, self).__init__()
        config = configparser.RawConfigParser()
        try:
            config.read(path)
        except Exception as err:
            logging.error(err)
            raise
        self._parse(config)

    def _parse(self, config):
        sections = config.sections()
        for section in sections:
            val = Valuer()
            options = config.options(section)
            for option in options:
                val[option] = config.get(section, option)
            self[section] = val

    def __repr__(self):
        return str(self._dict)

