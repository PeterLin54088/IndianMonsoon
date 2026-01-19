class StrictReadOnlyMeta(type):
    def __setattr__(cls, name, value):
        raise AttributeError(f"{cls.__name__} is fully read-only")