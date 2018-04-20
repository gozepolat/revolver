# -*- coding: utf-8 -*-
import uuid
from stacked.utils import common
from six import string_types
from six import reraise as raise_
from six import add_metaclass
import sys
from logging import warning


def log(log_func, msg):
    if common.DEBUG_SCOPE:
        log_func(msg)


def validate_scope(scope):
    if isinstance(scope, string_types):
        if len(scope) < 1:
            traceback = sys.exc_info()[2]
            raise_(ValueError, "Scope string can not be empty", traceback)
    else:
        traceback = sys.exc_info()[2]
        raise_(TypeError, "Scope must be a string", traceback)


class ScopedMeta(type):
    """A metaclass that creates a base Scoped class when called.

    '_random' is reserved for generating a random scope that will not be shared
    '~' is reserved for denoting the beginning of uuid, if instance is unique
    All instances will be stored in the common.SCOPE_DICTIONARY"""

    def __call__(cls, scope, *args, **kwargs):
        validate_scope(scope)
        if scope not in common.SCOPE_DICTIONARY:
            if scope == "_random":
                scope = generate_random_scope(scope)
                while scope in common.SCOPE_DICTIONARY:
                    scope = generate_random_scope(scope)

            # uniquely associate the scope with the generated instance
            instance = super(ScopedMeta, cls).__call__(scope, *args, **kwargs)
            common.SCOPE_DICTIONARY[scope] = {'meta': dict(), 'instance': instance}
        else:
            log(warning, "Scope {} already exists, "
            "ignoring the constructor arguments".format(scope))

        scoped = common.SCOPE_DICTIONARY[scope]
        scoped_instance = scoped['instance']
        scoped_type = type(scoped_instance)
        if scoped_type != cls:
            traceback = sys.exc_info()[2]
            error = "Same scope, different types: {}! \
            Current: {}, registered type: {}".format(scope, cls, scoped_type)
            raise_(TypeError, error, traceback)

        return scoped_instance


@add_metaclass(ScopedMeta)
class Scoped(object):
    def __init__(self, scope, *args, **kwargs):
        self.scope = scope


def get_instance(scope, *args, **kwargs):
    """Get or make shared class instance of any class, without the metamagic"""
    validate_scope(scope)
    if scope in common.SCOPE_DICTIONARY:
        return common.SCOPE_DICTIONARY[scope]['instance']

    def _make_element(element_scope, op, *args_tail, **rest):
        scoped_op = op(*args_tail, **rest)
        meta = dict()
        if hasattr(scoped_op, 'meta'):
            meta = scoped_op.meta
        common.SCOPE_DICTIONARY[element_scope] = {'meta': meta, 'instance': scoped_op}
        return scoped_op

    return _make_element(scope, *args, **kwargs)


def get_meta(scope):
    """Get meta information about the scope"""
    validate_scope(scope)
    if scope in common.SCOPE_DICTIONARY:
        return common.SCOPE_DICTIONARY[scope]['instance']
    else:
        raise_(KeyError, "scope {} not recorded".format(scope))


def get_elements():
    return common.SCOPE_DICTIONARY


def generate_random_scope(prefix=""):
    return '~'.join([prefix, uuid.uuid4().hex])

