# -*- coding: utf-8 -*-
import uuid
from revolver.utils import common
from six import string_types
from six import add_metaclass
import sys
from logging import warning
import torch

UNIQUE_SUFFIX_DELIMITER = '+'


def log(log_func, msg):
    if common.DEBUG_SCOPE:
        if common.LAST_DEBUG_MESSAGE != msg:
            log_func("revolver.meta.scope: %s" % msg)
            common.LAST_DEBUG_MESSAGE = msg


def validate_scope(scope):
    if isinstance(scope, string_types):
        if len(scope) < 1:
            raise ValueError("Scope string can not be empty")
    else:
        raise TypeError("Scope must be a string")


class ScopedMeta(type):
    """A metaclass that creates a scoped class when called.

    Args:
        scope (str): identifier for the scope name
        '_random' is reserved for generating a random scope that will not be shared
        '+' is reserved for denoting the beginning of uuid, if instance is unique

    All scoped class instances will be stored in the common.SCOPE_DICTIONARY
    """

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
            log(warning, "Scope {} already exists, \
            ignoring the constructor arguments".format(scope))

        scoped = common.SCOPE_DICTIONARY[scope]
        scoped_instance = scoped['instance']
        scoped_type = type(scoped_instance)
        if scoped_type != cls:
            traceback = sys.exc_info()[2]
            error = "Same scope, different types: {}! \
            Current: {}, registered type: {}".format(scope, cls, scoped_type)
            raise TypeError(error)

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


def unregister(scope):
    if scope in common.SCOPE_DICTIONARY:
        del common.SCOPE_DICTIONARY[scope]


def is_registered(scope):
    return scope in common.SCOPE_DICTIONARY


def get_meta(scope):
    """Get meta information about the scope"""
    validate_scope(scope)
    if scope in common.SCOPE_DICTIONARY:
        return common.SCOPE_DICTIONARY[scope]['meta']
    else:
        raise KeyError("scope {} not recorded".format(scope))


def get_elements():
    return common.SCOPE_DICTIONARY


def generate_random_scope(prefix=""):
    return UNIQUE_SUFFIX_DELIMITER.join([prefix, uuid.uuid4().hex])


def get_common_scope_name_length(scope):
    validate_scope(scope)
    return scope.find(UNIQUE_SUFFIX_DELIMITER)
