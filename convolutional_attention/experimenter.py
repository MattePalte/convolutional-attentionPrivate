
class ExperimentLogger(object):

    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        return

    def record_results(self, dict_metrics):
        print "Model name:"
        print self.model_name
        print "Model parameters:"
        print self.params
        print "Performance Metrics:"
        print dict_metrics