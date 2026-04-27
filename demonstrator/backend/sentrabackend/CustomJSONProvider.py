from flask.json.provider import DefaultJSONProvider

from .SentraNode import SentraNode
from .SentraNodeList import SentraNodeList
from .SentraML import InferenceResult,ClassificationResult

class CustomJSONProvider(DefaultJSONProvider):

    @staticmethod
    def default(obj:object)->object:  # type: ignore
        if isinstance(obj, SentraNode) or isinstance(obj,SentraNodeList) or isinstance(obj,InferenceResult)  or isinstance(obj,ClassificationResult):
            return obj.toJSONObject()
        return DefaultJSONProvider.default(obj)