from collections.abc import MutableMapping


class AliasedDict(MutableMapping):
    """Dict with aliasing and case insensitivity"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

        self.aliases = dict()

    def __getitem__(self, key):
        return self.store[self._get_alias(key)]

    def __setitem__(self, key, value):
        self.store[self._get_alias(key)] = value

    def __delitem__(self, key):
        del self.store[self._get_alias(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _get_alias(self, key):
        return key.lower()

    def add_alias(self, *aliases):
        
        aliases = [alias.lower() for alias in aliases]
        aliases = self.store.get(root.lower())
