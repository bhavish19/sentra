from flask.json.provider import DefaultJSONProvider

from .SentraNode import SentraNode
from .SentraNodeList import SentraNodeList

class CustomJSONProvider(DefaultJSONProvider):

    def default(self, obj:object)->object:
        if isinstance(obj, SentraNode) or isinstance(obj,SentraNodeList):
            return obj.toJSONObject()
        return super().default(obj)