import copy
import numpy as np


class Attribute:
    pass


class Data:

    def __init__(self, *, fpath="", data=None):

        if not fpath and data is None:
            raise Exception("Must pass either a path to a data file or a numpy array object")

        self.raw_data, self.attributes, self.index_column_dict, \
            self.column_index_dict = self._load_data(fpath, data)

    def _load_data(self, fpath="", data=None):

        if data is None:
            data = np.loadtxt(fpath, delimiter=',', dtype=str)

        header = data[0]
        index_column_dict = dict(enumerate(header))

        # Python 2.7.x
        # column_index_dict = {v: k for k, v in index_column_dict.items()}

        # Python 3+
        column_index_dict = {v: k for k, v in index_column_dict.items()}

        data = np.delete(data, 0, 0)

        attributes = self._set_attributes_info(index_column_dict, data)

        return data, attributes, index_column_dict, column_index_dict

    def _set_attributes_info(self, index_column_dict, data):
        attributes = dict()

        for index in index_column_dict:
            column_name = index_column_dict[index]
            if column_name == 'label':
                continue
            attribute = Attribute()
            attribute.name = column_name
            attribute.index = index - 1
            attribute.possible_vals = np.unique(data[:, index])
            attributes[column_name] = attribute

        return attributes

    def get_attribute_possible_vals(self, attribute_name):
        """

        Given an attribute name returns the all of the possible values it can take on.

        Args:
            attribute_name (str)

        Returns:
            TYPE: numpy.ndarray
        """
        return self.attributes[attribute_name].possible_vals

    def get_row_subset(self, attribute_name, attribute_value, data=None):
        """

        Given an attribute name and attribute value returns a row-wise subset of the data,
        where all of the rows contain the value for the given attribute.

        Args:
            attribute_name (str):
            attribute_value (str):
            data (numpy.ndarray, optional):

        Returns:
            TYPE: numpy.ndarray
        """
        if not data:
            data = self.raw_data

        column_index = self.get_column_index(attribute_name)
        new_data = copy.deepcopy(self)
        new_data.raw_data = data[data[:, column_index] == attribute_value]
        return new_data

    def get_column(self, attribute_names, data=None):
        """

        Given an attribute name returns the corresponding column in the dataset.

        Args:
            attribute_names (str or list)
            data (numpy.ndarray, optional)

        Returns:
            TYPE: numpy.ndarray
        """
        if not data:
            data = self.raw_data

        if type(attribute_names) is str:
            column_index = self.get_column_index(attribute_names)
            return data[:, column_index]

        column_indicies = []
        for attribute_name in attribute_names:
            column_indicies.append(self.get_column_index(attribute_name))

        return data[:, column_indicies]

    def get_column_index(self, attribute_name):
        """
        Given an attribute name returns the integer index that corresponds to it.
        Args:
            attribute_name (str)
        Returns:
            TYPE: int
        """
        return self.column_index_dict[attribute_name]

    def __len__(self):
        return len(self.raw_data)
