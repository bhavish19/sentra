from flask.json.provider import DefaultJSONProvider

from .SentraNode import SentraNode
from .SentraNodeList import SentraNodeList

class CustomJSONProvider(DefaultJSONProvider):

    @staticmethod
    def default(obj:object)->object:  # type: ignore
        if isinstance(obj, SentraNode) or isinstance(obj,SentraNodeList):
            return obj.toJSONObject()
        return DefaultJSONProvider.default(obj)