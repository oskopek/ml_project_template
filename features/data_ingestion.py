from os import listdir
from os.path import isdir, isfile, join
import matplotlib.image as mpimg


class DataFile(object):

    SUFFIXES = ['jpg', 'png']

    def __init__(self, base, file):
        self.file = base + '/' + file
        assert isdir(base)
        assert isfile(self.file)

        self.contents = None

    def is_loaded(self):
        assert self.file is not None
        return self.contents is not None

    def is_image(self, filename):
        for suffix in self.SUFFIXES:
            if filename.endswith(suffix):
                return True
        return False

    def load(self):
        assert self.file is not None
        if self.contents is not None:
            return

        # TODO: load the samples into memory
        if isfile(self.file):
            if self.file.endswith('.csv'):
                pass
            elif self.file.endswith('.npy'):
                pass
        elif isdir(self.file):
            onlyfiles = [f for f in listdir(self.file) if isfile(join(self.file, f))]
            onlyfiles = [f for f in onlyfiles if self.is_image(f)]
            for f in onlyfiles:
                mpimg.imread(f)

    def unload(self):
        assert self.file is not None
        if self.contents is None:
            return

        self.contents = None

    def split_train_dev_test(self):
        # TODO: split into train test dev, either file by file, or by loading on demand, or something
        # Needs labels for balanced split!
        # TODO: figure out how to do a balanced split across different datasets!
        pass


class DataSet(object):

    def __init__(self, base, file):
        self.train = DataFile(base, file + '_train.npz')
        self.test = DataFile(base, file + '_test.npz')
        self.dev = DataFile(base, file + '_dev.npz')
        self.file = base + '/' + file

    def load(self, which=None):
        pass  # TODO

    def unload(self, which=None):
        pass

    def is_loaded(self):
        # TODO
        pass

# class DataSets() TODO: Just use a dict.
