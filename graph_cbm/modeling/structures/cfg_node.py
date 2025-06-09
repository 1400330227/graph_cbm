class CfgNode(dict):
    def __init__(self, *args, **kwargs):
        super(CfgNode, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return CfgNode(value)
            return value
        except KeyError:
            raise AttributeError(f"'CfgNode' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value
