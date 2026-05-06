import threading
import random
import time

import numpy as np
import tensorflow as tf

from .SentraNodeAttributeGenerator import SentraNodeAttributeGenerator
from .SentraNodeList import SentraNodeList
from .SentraNode import SentraNode
from .ComitteeSelection import CommitteeSelection
from .Log import log as log
from .SentraML import InferenceResult
class AppSimulator:

    m_Thread:threading.Thread|None
    m_nodeGenerator:SentraNodeAttributeGenerator
    m_Model=None
    m_nodeList:SentraNodeList
    m_committeeSelection:bool


    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList, committeeSelection:bool):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_committeeSelection = committeeSelection
        threadTraining = threading.Thread(target=self.training, args=(), daemon=True)
        threadTraining.start()


    def runSimulation(self):
  
        from .Backend import Backend
        baseId:str="SentraNode_"
        baseLabel:str="Node "
        while(True):
            i:int=0
            self.m_nodeList.clear()     
            self.m_nodeGenerator.shuffleOperators()          
            while(i<10):
                node:SentraNode=self.m_nodeGenerator.generateNode(baseId+str(i),baseLabel+str(i))
                if(random.random()>0.3):
                    node.setVerified(True, time.time())
                self.m_nodeList.add(node)
                i+=1

            if self.m_committeeSelection:
                log("node list:")
                log(str(self.m_nodeList))

                committee_target_size = 5
                committee_min_trust = 2
                committee_max_attest_age = 100
                committee_max_hw_frac = 0.8
                committee_max_op_frac = 0.8
                committee_max_pm_frac = 0.8

                committeeSelection: CommitteeSelection = CommitteeSelection(committee_target_size, committee_min_trust, committee_max_attest_age, committee_max_hw_frac, committee_max_op_frac, committee_max_pm_frac)

                committee = committeeSelection.selectionAlgorithm(self.m_nodeList)

                log(f"running committee selction with: committee target size: {committee_target_size}, committee min trust: {committee_min_trust}, committee_max_attest_age: {committee_max_attest_age}, committee_max_hw_frac: {committee_max_hw_frac}, committee_max_op_frac: {committee_max_op_frac}, committee_max_pm_frac: {committee_max_pm_frac}")

                if committee:
                    log("committee:")
                    log(str(committee))
                    Backend.getBackend().setCommittee(committee)
                    break
                else:
                    log("no committee found! - Restarting")
        Backend.getBackend().notifyNodeListUpdated()


    def start(self):
        self.m_Thread = threading.Thread(target=self.runSimulation, args=(), daemon=True)
        self.m_Thread.start()

    def restart(self):
        self.start()

    def doInferenceForPixel(self,pixels)->InferenceResult:
        pixel_array = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1)  # Reshape to [1, 28, 28, 1]
      #  pixel_array /= 255.0
        predicted_probabilities = self.m_Model.predict(pixel_array)  # Shape: (1, 10)
        predicted_class = tf.argmax(predicted_probabilities, axis=1).numpy()[0]

        # Print results
        print(f"Predicted Class: {predicted_class}")
        print(f"Predicted Probabilities: {predicted_probabilities}")
        return InferenceResult(predicted_class,predicted_probabilities[0].tolist())

    def doInference(self,inputImageIndex:int)->InferenceResult:
        example_image = self.x_test[inputImageIndex:inputImageIndex + 1]  # Shape: (1, 28, 28, 1)
        actual_label = self.y_test[inputImageIndex]

        predicted_probabilities = self.m_Model.predict(example_image)  # Shape: (1, 10)
        predicted_class = tf.argmax(predicted_probabilities, axis=1).numpy()[0]

        # Print results
        print(f"Actual Label: {actual_label}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Predicted Probabilities: {predicted_probabilities}")
        return InferenceResult(predicted_class,predicted_probabilities[0].tolist())
  

    def loadMNISTDataset(self):

        # Load datasets
        data = np.load('resources/mnist.npz')
        x_train2, y_train2 = data['x_train'], data['y_train']
        x_test2, y_test2 = data['x_test'], data['y_test']
        return (x_train2,y_train2),(x_test2,y_test2)



    def training(self):
        (x_train, y_train), (x_test, self.y_test) = self.loadMNISTDataset()
        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32")  / 255.0
        x_train = x_train[..., tf.newaxis]   # shape: (60000, 28, 28, 1)
        self.x_test  = x_test[..., tf.newaxis]    # shape: (10000, 28, 28, 1)
        self.m_Model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
        ], name="mnist_cnn")

        self.m_Model.summary()
        self.m_Model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.m_Model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )
  
