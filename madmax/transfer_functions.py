from madmax.statistics import ProbabilityDensityFunction

class TransferFunction(object):
    __slots__ = ['parameter', 'pdf']
    def __init__(self, parameter, pdf):
        if not isinstance(pdf, ProbabilityDensityFunction):
            raise ValueError("pdf should be a (sub-class of) ProbabilityDensityFunction")
        self.parameter = parameter
        self.pdf = pdf
