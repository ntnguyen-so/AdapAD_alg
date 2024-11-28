import config

class NormalDataPredictionErrorCalculator():
    @staticmethod
    def calc_error(ground_truth, predict):
        return (ground_truth - predict)**2
        
class NormalValueRangeDb():
    def __init__(self):
        _, self.__value_range, _ = config.init_config()
        
    def lower(self):
        return self.__value_range[0]
        
    def upper(self):
        return self.__value_range[1]
        
class PredictedNormalDataDb():
    def __init__(self):
        self.predicted_vals = list()
        
    def append(self, val):
        self.predicted_vals.append(val)
    
    def clean(self, keep_length):
        self.predicted_vals = self.predicted_vals[-keep_length:]
        
    def get_tail(self, length=1):
        if length == 1:
            return self.predicted_vals[-1]
        else:
            return self.predicted_vals[-length:]
            
    def get_length(self):
        return len(self.predicted_vals)
        
    def clean(self, len2keep):
        self.predicted_vals = self.predicted_vals[-len2keep:]
        
class AnomalousThresholdDb():
    def __init__(self):
        self.thresholds = list()
        
    def append(self, val):
        self.thresholds.append(val)
    
    def clean(self, keep_length):
        self.thresholds = self.thresholds[-keep_length:]
        
    def get_tail(self, length=1):
        if length == 1:
            return self.thresholds[-1]
        else:
            return self.thresholds[-length:]
            
    def get_length(self):
        return len(self.thresholds)
        
    def clean(self, len2keep):
        self.thresholds = self.thresholds[-len2keep:]
        
class PredictionErrorDb():
    def __init__(self, prediction_error_training):
        self.prediction_errors = prediction_error_training.copy()
        
    def append(self, val):
        self.prediction_errors.append(val)
    
    def clean(self, keep_length):
        self.prediction_errors = self.prediction_errors[-keep_length:]
        
    def get_tail(self, length=1):
        if length == 1:
            return self.prediction_errors[-1]
        else:
            return self.prediction_errors[-length:]
    
    def get_length(self):
        return len(self.prediction_errors)
        
    def clean(self, len2keep):
        self.prediction_errors = self.prediction_errors[-len2keep:]
        
class DataSubject():
    def __init__(self, normal_data):
        self.observed_vals = normal_data.copy()
        self.is_retrieved_training_data = False
    
    def get_tail(self, length=1):
        if length == 1:
            return self.observed_vals[-1]
        else:
            return self.observed_vals[-length:]
        
    def append(self, val):
        self.observed_vals.append(val)
        
    def get_training_data(self):
        if self.is_retrieved_training_data:
            raise Exception("Already retrieved training data! Check the flow...")
        
        self.is_retrieved_training_data = True
        return self.observed_vals
        
    def get_length(self):
        return len(self.observed_vals)