from stacked.meta.scoped import generate_random_scope
from stacked.utils.transformer import all_to_none


class Blueprint(dict):
    """Dictionary like description of a scoped module

    Args:
        prefix (str): The root scope name of the module
        unique (bool): Indication of uniqueness, if True
                       append a random string to the scope name
        module_type (type): Type that will be used as constructor;
                            if not given, the constructor will return None
        args (iterable): Arguments that will be used in the constructor
        children (iterable): Member module descriptions
        description (dict): Dictionary form of the whole description
    """

    def __init__(self, prefix='None', unique=False,
                 module_type=all_to_none, args=[], children=[], description={}):
        super(Blueprint, self).__init__(description)

        # set from args if not in description
        if 'name' not in self:
            self['name'] = prefix
        if 'args' not in self:
            self['args'] = args
        if 'type' not in self:
            self['type'] = module_type
        if 'unique' not in self:
            self['unique'] = unique
        if 'prefix' not in self:
            self['prefix'] = prefix
        if 'children' not in self:
            self['children'] = children

        if self['unique']:
            self.make_unique()

    def make_unique(self):
        self['unique'] = True
        self['name'] = "%s-%s" % (self['prefix'], generate_random_scope())
