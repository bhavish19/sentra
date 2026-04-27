class ClassificationResult:
    predictedValue:int
    likelihood:float

    def __init__(self,v:int,l:float):
        self.predictedValue=int(v)
        self.likelihood=l    

    def toJSONObject(self)->object:
        return {
            'predictedValue':self.predictedValue,
            'likelihood':self.likelihood
        }
    
class InferenceResult:
    m_predictedValue:int
    m_classificationResults:list[ClassificationResult]

    def __init__(self,predictedValue:int,likelihoods:list[float]):
        self.m_predictedValue=int(predictedValue)
        self.m_classificationResults=[]
        i:int=0
        while(i<10):
            c:ClassificationResult=ClassificationResult(i,likelihoods[i])
            self.m_classificationResults.append(c)
            i=i+1

    def toJSONObject(self)->object:
        return {
            'predictedValue':self.m_predictedValue,
            'classificationResults':self.m_classificationResults
        }
    




