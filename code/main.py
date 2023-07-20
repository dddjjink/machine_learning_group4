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
    dataset_factory = DataFactory()
    _dataset_ = dataset_factory.create_dataset("iris")
    _dataset_.data_target()
    # print(_dataset_.data)
    X = _dataset_.data
    y = _dataset_.target
    splitter_factory = SplitterFactory()
    _splitter_ = splitter_factory.create_splitter("cv", X, y)

    X_train, X_test, y_train, y_test = _splitter_.split()
