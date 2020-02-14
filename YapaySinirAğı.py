from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, a, b):
        self.weights = 2 * random.random((a, b)) - 1


class ArtificalNeuralNetwork():
    def __init__(self, hiddenLayer, lastLayer):
        self.hiddenLayer = hiddenLayer
        self.lastLayer = lastLayer

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, trainingSetInputs):
        hiddenLayyerOutput = self.sigmoid(dot(trainingSetInputs, self.hiddenLayer.weights))
        lastLayerOutput = self.sigmoid(dot(hiddenLayyerOutput, self.lastLayer.weights))
        return hiddenLayyerOutput, lastLayerOutput

    def back_propagation(self, trainingSetInputs, trainingSetOutputs, epoch ,learningRate):
        for iter in range(epoch):
            hiddenLayyerOutput, lastLayerOutput = self.forward_propagation(trainingSetInputs)

            lastLayerError = trainingSetOutputs - lastLayerOutput
            lastLayerDelta = lastLayerError * self.sigmoid_derivative(lastLayerOutput)

            hiddenLayerError = lastLayerDelta.dot(self.lastLayer.weights.T)
            hiddenLayerDelta = hiddenLayerError * self.sigmoid_derivative(hiddenLayyerOutput)

            hiddenLayerAdjustment = trainingSetInputs.T.dot(hiddenLayerDelta)
            lastLayerAdjustment = hiddenLayyerOutput.T.dot(lastLayerDelta)

            self.hiddenLayer.weights += hiddenLayerAdjustment * learningRate
            self.lastLayer.weights += lastLayerAdjustment * learningRate



if __name__ == "__main__":
    random.seed(1)

    number = int(input("Ara nöron sayisi=>"))

    hiddenLayer = NeuronLayer(4, number)

    lastLayer = NeuronLayer(number, 2)

    ann = ArtificalNeuralNetwork(hiddenLayer, lastLayer)

    trainingSetInputs= array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    trainingSetOutputs = array([[0, 0, 1, 1], [0, 1, 0, 1]]).T

    epoch =int(input("Epoch sayisi =>"))
    ann.back_propagation(trainingSetInputs, trainingSetOutputs, epoch, learningRate=0.1)

    print("örn: 0, 1, 0, 0  için sonuç =>")
    hidden_state, output = ann.forward_propagation(array([0, 1, 0, 0]))
    print(output)