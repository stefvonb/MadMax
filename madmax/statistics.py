
import numpy as np
import math
import scipy
from scipy.integrate import quad

class Model(object):
    def __init__(self, **kwargs):
        self.distributions = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) == 2:
                self.distributions[k] = {'pdf': v[0], 'starting_value': v[1]}
            else:
                print("WARNING:\tCould not add {} to model. Please ensure it is of type [pdf, starting value]".format(v))

    def add_pdf(self, key, pdf, starting_value):
        if key in self.distributions:
            print("WARNING:\tPDF with key {} already in model. Replacing...".format(key))
        self.distributions[key] = {'pdf': pdf, 'starting_value': starting_value}

    def __getitem__(self, key):
        return self.distributions[key]
    
    def __setitem__(self, key, value):
        self.add_pdf(key, value[0], value[1])

    def cube_to_standard(self, input_list, order):
        if isinstance(order, list):
            if len(input_list) == len(order):
                return [self.distributions[i]['pdf'].cube_to_standard(c) for i, c in zip(order, input_list)]
            else:
                print("ERROR:\tInput list and order should have the same length")
                return None
        else:
            print("ERROR:\tShould take a list of the order of PDFs as an input")
            return None

    def standard_to_cube(self, input_list, order):
        if isinstance(order, list):
            if len(input_list) == len(order):
                return [self.distributions[i]['pdf'].standard_to_cube(c) for i, c in zip(order, input_list)]
            else:
                print("ERROR:\tInput list and order should have the same length")
                return None
        else:
            print("ERROR:\tShould take a list of the order of PDFs as an input")
            return None

    def sample(self, order):
        if isinstance(order, list):
            return [self.distributions[i]['pdf'].sample() for i in order]
        else:
            print("ERROR:\tSample should take a list of the order of PDFs as an input")
            return None

    def starting_values(self, order):
        if isinstance(order, list):
            return [self.distributions[i]['starting_value'] for i in order]
        else:
            print("ERROR:\tShould take a list of the order of PDFs as an input")
            return None

    def cube_starting_values(self, order):
        return self.standard_to_cube(self.starting_values(order), order)


class ProbabilityDensityFunction(object):
    """
    This is a 'template' class for PDFs
    Each sub-clas should define:
    - f: the normalised functional form of the PDF with one variable
    - F: the CDF, which integrates f up until some value x
    - inv_F: the inverse of the CDF, which samples the PDF at random inputs between 0 and 1
    """
    def f(self, x):
        return None

    def F(self, x):
        return None

    def inv_F(self, F):
        return None

    def __getitem__(self, x):
        return self.f(x)

    def standard_to_cube(self, standard_coordinate):
        return self.F(standard_coordinate)
    
    def cube_to_standard(self, cube_coordinate):
        return self.inv_F(cube_coordinate)

    def sample(self):
        return self.cube_to_standard(np.random.uniform())


class BreitWigner(ProbabilityDensityFunction):
    __slots__ = ['_k', '_m02', '_width2', '_m0width']
    def __init__(self, mass, width):
        self._m02 = math.pow(mass, 2)
        self._width2 = math.pow(width, 2)
        gamma = math.sqrt(self._m02*(self._m02 + self._width2))
        self._k = math.sqrt(8)/math.pi*(mass*width*gamma)/(math.sqrt(self._m02 + gamma))
        self._m0width = mass*width

    @property
    def mass(self):
        return math.sqrt(self._m02)

    @mass.setter
    def mass(self, mass):
        width = self._m0width/self.mass
        self._m02 = math.pow(mass, 2)
        gamma = math.sqrt(self._m02*(self._m02 + self._width2))
        self._k = math.sqrt(8)/math.pi*(mass*width*gamma)/(math.sqrt(self._m02 + gamma))
        self._m0width = mass*width

    @property
    def width(self):
        return self._m0width/self.mass

    @width.setter
    def width(self, width):
        self._width2 = math.pow(width, 2)
        gamma = math.sqrt(self._m02*(self._m02 + self._width2))
        self._k = math.sqrt(8)/math.pi*(self.mass*width*gamma)/(math.sqrt(self._m02 + gamma))
        self._m0width = self.mass*width

    def f(self, x):
        return self._k/(math.pow(math.pow(x, 2) - self._m02, 2) + self._m02*self._width2)

    def F(self, x):
        m2 = math.pow(x, 2)
        return 1/math.pi*(math.atan((m2 - self._m02)/self._m0width) + math.pi/2)

    def inv_F(self, F):
        m2 = self._m02 + self._m0width*math.tan(math.pi*F - math.pi/2)
        if m2 < 0.0:
            return 0.0
        else:
            return math.sqrt(m2)  


class Flat(ProbabilityDensityFunction):
    __slots__ = ['_lower_limit', '_upper_limit', '_height', '_width']
    def __init__(self, lower_limit, upper_limit):
        if lower_limit > upper_limit:
            raise ValueError("Lower limit should be less than upper limit")
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._width = upper_limit - lower_limit
        self._height = 1/self._width

    @property
    def lower_limit(self):
        return self._lower_limit

    @lower_limit.setter
    def lower_limit(self, l):
        self._lower_limit = l
        self._width = self._upper_limit - l
        self._height = 1/self._width

    @property
    def upper_limit(self):
        return self._upper_limit

    @upper_limit.setter
    def upper_limit(self, u):
        self._upper_limit = u
        self._width = u - self._lower_limit
        self._height = 1/self._width

    def f(self, x):
        if self._lower_limit < x < self._upper_limit:
            return self._height
        else:
            return 0.0

    def F(self, x):
        return (x - self._lower_limit)/self._width

    def inv_F(self, F):
        return self._lower_limit + F*self._width


class Gaussian(ProbabilityDensityFunction):
    __slots__ = ['_mean', '_sigma', '_variance', '_amplitude']
    def __init__(self, mean, sigma):
        self._mean = mean
        self._sigma = sigma
        self._variance = math.pow(sigma, 2)
        self._amplitude = 1/math.sqrt(2*math.pi*self._variance)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        self._variance = math.pow(sigma, 2)
        self._amplitude = 1/math.sqrt(2*math.pi*self._variance)

    def f(self, x):
        return self._amplitude*math.exp(-(x - self._mean)**2/(2*self._variance))

    def F(self, x):
        return 0.5*(1 + math.erf((x - self._mean)/(math.sqrt(2)*self._sigma)))

    def inv_F(self, F):
        return self._mean + math.sqrt(2)*self._sigma*scipy.special.erfinv(2*F - 1)


class BreitWignerOffshell(ProbabilityDensityFunction):
    @staticmethod
    def f_shoulder_prenorm(m, w, m_parent, m_sibling):
        beta_v = math.sqrt((1 - math.pow((m_sibling + m)/m_parent, 2))*(1 - math.pow((m_sibling - m)/m_parent, 2)))
        return (beta_v*(m_parent**4*beta_v**2 + 12*m_sibling**2*m**2))/(math.pow(m**2 - m_sibling**2, 2) + m_sibling**2*w**2)

    __slots__ = ['_mass', '_width', '_m_parent', '_m_sibling', '_peak', '_is_offshell', '_k_shoulder']
    def __init__(self, mass, width, parent_mass, sibling_mass):
        self._mass = mass
        self._width = width
        self._m_parent = parent_mass
        self._m_sibling = sibling_mass
        self._is_offshell = False
        self._peak = BreitWigner(mass, width)

        if parent_mass < mass + sibling_mass:
            self._is_offshell = True
        if self._is_offshell:
            self._peak._k *= 0.5
            integral = quad(self.f_shoulder_prenorm, 0.0, parent_mass - sibling_mass, args=(width, parent_mass, sibling_mass))
            self._k_shoulder = 0.5/integral[0]

    def f_shoulder(self, x):
        if self._is_offshell and 0.0 < x < self._m_parent - self._m_sibling:
            return self._k_shoulder*self.f_shoulder_prenorm(x, self._width, self._m_parent, self._m_sibling)
        else:
            return 0.0

    def f(self, x):
        if not self._is_offshell:
            return self._peak.f(x)
        else:
            if x < self._m_parent - self._m_sibling:
                return self.f_shoulder(x)
            else:
                return self._peak.f(x)

    def F(self, x):
        if not self._is_offshell:
            return self._peak.F(x)
        else:
            if x < self._m_parent - self._m_sibling:
                return quad(self.f_shoulder, 0.0, x)[0]
            else:
                return 0.5*self._peak.F(x) + 0.5

    def inv_F(self, F):
        if not self._is_offshell:
            return self._peak.inv_F(F)
        else:
            if F < 0.5:
                return 0.0#return quad(self.f_shoulder, 0.0, x)[0]
            else:
                return self._peak.inv_F(0.5*F)
    