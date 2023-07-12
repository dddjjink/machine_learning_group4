class DataFactory:
    def __init__(self) -> None:
        self.elements = dict()

    def register(self, name, element):
        self.elements.setdefault(name, element)

    def inspect(self):
        return self.elements.keys()

    def getData(self, name: str):
        # if name == 'iris':
        #     return IrisDataset
        # elif name == 'house':
        #     return HouseDataset
        # else:
        #     raise NotImplementedError()
        return self.elements[name]


class SplitterFactory:
    pass


class ModelFactory:
    pass


class EvaluationFactory:
    pass


if __name__ == '__main__':
    # data_name = 'iris'
    data_name = 'house'

    data_factory = DataFatory()

    data_factory.register('iris', IrisDataset)
    data_factory.register('house', HouseDataset)

    result = data_factory.inspect()

    data = data_factory.getData(data_name)
    data.load()
    splitter = Splitter()
    training_set, test_set, validation_set = splitter.split(data)

    ...
