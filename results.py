# Results are organized with data and metadata
# data is a pandas DataFrame with multiindex to organize experiment -> positions -> single cells
# metadata is a pandas DataFrame with a unique mapping between rows of data and rows of a 
#   pandas DataFrame containing columns of metadata about the data entries.
#
#
#
#
#
from pandas import DataFrame

class Results(object):
    def __init__(self, pth=None):
        self.path = pth
        self.Conclusions = 'missing'
        self.analysisScript = ''
        self.reportScript = ''
        
        self.Data = DataFrame()
        self.EntryMetadata = DataFrame()
        self.levels = ['experiment']
    
    def saveResults(self, pth):
        raise ValueError('Not implemented')
        
    def publish(self, varargin):
        raise ValueError('Not implemented')
        
    def add(self, newdata, varargin):
        raise ValueError('Not implemented')
        
    def deleteData(self, data_name):
        raise ValueError('Not implemented')
        
    def getData(self, data_name):
        raise ValueError('Not implemented')
        
    
class MultiPositionSingleCellResults(Results):
    def __init__(self, pth=None):
        self.levels = ['base', 'pos', 'cell']
    
    def add(self, data, pos, base=None, metadata=None):
        """
        Add new data to the results object.
        
        Parameters
        ----------
        data : numpy.array, list, tuple, dict
        pos : str
        base : str
        
        Supported Input Styles
        --------- ----- ------
        2D ndarray
        list
        
        """