# -*- coding: utf-8 -*-
import uuid
from future.utils import raise_with_traceback
from stacked.utils import common


def validate_scope(scope):
    if isinstance(scope, basestring):
        if len(scope) < 1:
            raise_with_traceback(ValueError("Scope string can not be empty"))
    else:
        raise_with_traceback(TypeError("Scope must be a string"))


class _Scoped(type):
    """A metaclass that creates a base Scoped class when called.

    '_random' is reserved for generating a random scope that will not be shared
    All instances will be stored in the common.SCOPE_DICTIONARY"""

    def __call__(cls, scope, meta, *args, **kwargs):
        validate_scope(scope)
        if scope not in common.SCOPE_DICTIONARY:
            if scope == "_random":
                scope = generate_random_scope(scope)
                while scope in common.SCOPE_DICTIONARY:
                    scope = generate_random_scope(scope)

            # uniquely associate the scope with the generated instance
            common.SCOPE_DICTIONARY[scope] = {'meta': meta,
                                       'instance': super(_Scoped, cls).__call__(scope, meta, *args, **kwargs)}

        scoped = common.SCOPE_DICTIONARY[scope]
        scoped_instance = scoped['instance']
        scoped_type = type(scoped_instance)

        if scoped_type != cls:
            raise_with_traceback(TypeError(
                "Same scope, different types: {}! Current: {}, registered type: {}".format(scope, cls, scoped_type)))

        return scoped_instance


class Scoped(_Scoped('ScopedMeta', (object,), {})):
    def __init__(self, scope, meta, *args, **kwargs):
        self.scope = scope
        self.meta = meta


def get_instance(scope, *args, **kwargs):
    """Get or make shared class instance of any class, without the metamagic"""
    validate_scope(scope)
    if scope in common.SCOPE_DICTIONARY:
        return common.SCOPE_DICTIONARY[scope]['instance']

    def _make_element(element_scope, op, *args_tail, **rest):
        scoped_op = op(*args_tail, **rest)
        meta = None
        if hasattr(scoped_op, 'meta'):
            meta = scoped_op.meta
        common.SCOPE_DICTIONARY[element_scope] = {'meta': meta, 'instance': scoped_op}
        return scoped_op

    return _make_element(scope, *args, **kwargs)


def get_elements():
    return common.SCOPE_DICTIONARY


def generate_random_scope(prefix=""):
    return '/'.join([prefix, uuid.uuid4().hex])

