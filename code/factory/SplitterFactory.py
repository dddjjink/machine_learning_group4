from Factory.Factory import Factory
from splitter.BootStrapping import BootStrapping
from splitter.CV import CV
from splitter.HoldOut import HoldOut


class SplitterFactory(Factory):

    @staticmethod
    def create_splitter(splitter, dataset):
        if splitter == 'bootstrap':
            return BootStrapping(dataset)
        elif splitter == 'cv':
            return CV(dataset)
        elif splitter == 'holdout':
            return HoldOut(dataset)
