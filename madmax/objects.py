import math
import numpy as np
from . import FP_TOLERANCE, PhysicsError
from itertools import permutations, chain, product
from copy import deepcopy

class ThreeVector(object):
    __slots__ = ['v1', 'v2', 'v3']
    def __init__(self, v1=None, v2=None, v3=None):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    @property
    def x(self):
        return self.v1

    @x.setter
    def x(self, value):
        self.v1 = value

    @property
    def y(self):
        return self.v2

    @y.setter
    def y(self, value):
        self.v2 = value

    @property
    def z(self):
        return self.v3

    @z.setter
    def z(self, value):
        self.v3 = value

    def __str__(self):
        return "({}, {}, {})".format(self.v1, self.v2, self.v3)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self.__class__(self.v1+other.v1, self.v2+other.v2, self.v3+other.v3)

    def __mul__(self, other):
        return self.v1*other.v1 + self.v2*other.v2 + self.v3*other.v3

    def mag2(self):
        return self*self

    @property
    def mag(self):
        return self.mag2()**0.5

    @property
    def theta(self):
        return math.acos(self.v3/(self.v1**2 + self.v2**2 + self.v3**2)**0.5)

    @property
    def phi(self):
        """
        Using the convention that phi goes from -pi to pi
        """
        phi = math.atan(self.v2/self.v1)
        if self.v1 < 0.0:
            phi += (-1)**(int(self.v2 < 0.0))*math.pi
        return phi

    @classmethod
    def from_mag_and_angles(cls, mag, theta, phi):
        v1 = mag*math.sin(theta)*math.cos(phi)
        v2 = mag*math.sin(theta)*math.sin(phi)
        v3 = mag*math.cos(theta)
        return cls(v1, v2, v3)

    def delta_phi(self, other):
        d_phi = abs(self.phi - other.phi)
        return min([d_phi, 2*math.pi - d_phi])

    @property
    def array(self):
        return np.array([self.v1, self.v2, self.v3], dtype=np.float)


class FourVector(ThreeVector):
    __slots__ = ['v0']
    def __init__(self, v0=None, v1=None, v2=None, v3=None):
        self.v0 = v0
        super().__init__(v1, v2, v3)
    
    @property
    def t(self):
        return self.v0

    @t.setter
    def t(self, value):
        self.v0 = value

    def __str__(self):
        return "({}, {}, {}, {})".format(self.v0, self.v1, self.v2, self.v3)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self.__class__(self.v0+other.v0, self.v1+other.v1, self.v2+other.v2, self.v3+other.v3)

    def __sub__(self, other):
        return self.__class__(self.v0-other.v0, self.v1-other.v1, self.v2-other.v2, self.v3-other.v3)

    def __eq__(self, other):
        if abs(self.v0 - other.v0) < FP_TOLERANCE and abs(self.v1 - other.v1) < FP_TOLERANCE \
        and abs(self.v2 - other.v2) < FP_TOLERANCE and abs(self.v3 - other.v3) < FP_TOLERANCE:
            return True
        else:
            return False

    def __mul__(self, other):
        return self.v0*other.v0 - super().__mul__(other)
 
    def three_vector(self):
        return ThreeVector(self.v1, self.v2, self.v3)

    @classmethod
    def from_mag_and_angles(cls, v0, mag, theta, phi):
        v1 = mag*math.sin(theta)*math.cos(phi)
        v2 = mag*math.sin(theta)*math.sin(phi)
        v3 = mag*math.cos(theta)
        return cls(v0, v1, v2, v3)

    @property
    def array(self):
        return np.array([self.v0, self.v1, self.v2, self.v3], dtype=np.float)


class FourMomentum(FourVector):
    def __init__(self, E=None, px=None, py=None, pz=None):
        super().__init__(E, px, py, pz)

    @property
    def E(self):
        return self.v0

    @E.setter
    def E(self, value):
        self.v0 = value

    @property
    def px(self):
        return self.v1

    @px.setter
    def px(self, value):
        self.v1 = value

    @property
    def py(self):
        return self.v2

    @py.setter
    def py(self, value):
        self.v2 = value

    @property
    def pz(self):
        return self.v3

    @pz.setter
    def pz(self, value):
        self.v3 = value

    @property
    def m2(self):
        return self.mag2()

    @property
    def m(self):
        if self.m2 < -FP_TOLERANCE:
            raise PhysicsError("Four momentum returned imaginary mass")
        elif self.m2 < 0.0:
            return 0.0
        else:
            return self.m2**0.5

    @property
    def pT(self):
        return math.sqrt(math.pow(self.v1, 2) + math.pow(self.v2, 2))

    def mass(self):
        return self.m

    def rapidity(self):
        return 0.5*math.log((self.E + self.pz)/(self.E - self.pz))

    def pseudorapidity(self):
        return 0.5*math.log((self.three_vector().mag + self.pz)/(self.three_vector().mag - self.pz))

    def scale_energy(self, factor, preserve_mass=True):
        if preserve_mass:
            try:
                p3_scaling = math.sqrt((factor**2 - 1)*(self.v0**2)/(self.v0**2 - self.m**2) + 1)
            except ValueError:
                raise PhysicsError("Energy scaling can not preserve mass of particle")
            self.v1 *= p3_scaling
            self.v2 *= p3_scaling
            self.v3 *= p3_scaling
        self.v0 *= factor

    def set_energy_preserve_mass(self, E):
        original_energy = self.E
        factor = E/original_energy
        preserved_mass = self.m
        self.E = E
        try:
            p3_scaling = math.sqrt((factor**2 - 1)*(original_energy**2)/(original_energy**2 - preserved_mass**2) + 1)
        except ValueError:
            raise PhysicsError("Energy scaling can not preserve mass of particle")
        self.v1 *= p3_scaling
        self.v2 *= p3_scaling
        self.v3 *= p3_scaling


class Particle(FourMomentum):
    """
    The class for a particle
    pid: the pdg code for the particle, 0 is unknown
    status: the status for determining event properties
        0 = unknown
        1 = invisible
        2 = visible
    """
    __slots__ = ['pid', 'status']
    def __init__(self, pid, status, four_momentum):
        super().__init__(four_momentum.v0, four_momentum.v1, four_momentum.v2, four_momentum.v3)
        self.pid = pid
        self.status = status

    @property
    def four_momentum(self):
        return FourMomentum(self.v0, self.v1, self.v2, self.v3)

    @four_momentum.setter
    def four_momentum(self, fm):
        self.v0 = fm.v0
        self.v1 = fm.v1
        self.v2 = fm.v2
        self.v3 = fm.v3


class Event(object):
    __slots__ = ['sqrt_s', 'x1', 'x2', 'particles', 'checkpoint_particles']
    def __init__(self, sqrt_s, x1=None, x2=None, **kwargs):
        self.sqrt_s = sqrt_s
        self.x1 = x1
        self.x2 = x2
        self.particles = {}
        self.checkpoint_particles = {}
        for k, v in kwargs.items():
            assert(isinstance(v, Particle)), \
            "Events constructed with keywords must be of type string: Particle"
            self.add_particle(k, v)

    def save_checkpoint(self):
        for k, v in self.particles.items():
            self.checkpoint_particles[k] = deepcopy(v)

    def load_checkpoint(self):
        for k, v in self.checkpoint_particles.items():
            self.particles[k] = deepcopy(v)

    def reset_particles(self, particle_list):
        for k in particle_list:
            self.particles[k] = deepcopy(self.checkpoint_particles[k])

    def add_particle(self, key, particle):
        self.particles[key] = particle
        self.checkpoint_particles[key] = None

    def get_momentum(self, key):
        return self.particles[key].four_momentum
    
    def set_momentum(self, key, momentum):
        self.particles[key].four_momentum = momentum

    def is_visible(self, key):
        return self.particles[key].status > 1

    def __str__(self):
        return_string = "Event config: sqrt_s={} GeV, x1={}, x2={}, ID={}".format(self.sqrt_s, self.x1, self.x2, id(self))
        for k in sorted(self.particles):
            return_string += "\n{}: PID={}\tStatus={}\tID={}\tMomentum={}".format(k, self.particles[k].pid, 
                self.particles[k].status, id(self.particles[k]), self.particles[k].four_momentum)
        return_string = return_string.replace("None", "?")
        return return_string

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, key):
        return self.particles[key]

    def __setitem__(self, key, value):
        self.particles[key] = value

    def momentum_matrix(self, ordered_keys):
        matrix = []
        matrix.append(np.array([self.x1*self.sqrt_s/2, 0.0, 0.0, self.x1*self.sqrt_s/2], dtype=np.float))
        matrix.append(np.array([self.x2*self.sqrt_s/2, 0.0, 0.0, -1*self.x2*self.sqrt_s/2], dtype=np.float))
        for k in ordered_keys:
            matrix.append(self[k].array)
        return np.array(matrix)

    def momentum_matrix_transpose(self, ordered_keys):
        return np.matrix.transpose(self.momentum_matrix(ordered_keys))

    def HT_div_2(self):
        return sum([p.pT for k, p in self.particles.items()])/2

    def make_permutations(self, squash_list=None):
        """
        Particles that should be permuted must have a forward slash in their key.
        Squash list will remove redundant permutations. List of lists (or tuples).

        Example, if you have a h->bb decay, you might not want to consider permutations of those b's.
        """
        permutations_dict = {}
        for key in self.particles:
            if "/" not in key:
                continue
            key_segs = key.split("/")
            if len(key_segs) != 2:
                raise ValueError("Ambiguous intention to permute particle with key = {}".format(key)) 
            if key_segs[0] not in permutations_dict:
                permutations_dict[key_segs[0]] = [key]
            else:
                permutations_dict[key_segs[0]].append(key)
        groups = [v for k, v in permutations_dict.items()]
        mappings = [list(chain.from_iterable(p)) for p in product(*map(permutations, groups))]

        if squash_list:
            remove_mappings = set()
            for s in squash_list:
                squash_indices = [mappings[0].index(p) for p in s]
                match_indices = [n for n in range(len(mappings[0])) if n not in squash_indices]
                for i, m_i in enumerate(mappings):
                    if i in remove_mappings:
                        continue
                    keep = list(map(lambda x: m_i[x], match_indices))
                    for j in range(i + 1, len(mappings)):
                        if j in remove_mappings:
                             continue
                        if list(map(lambda x: mappings[j][x], match_indices)) == keep:
                            remove_mappings.add(j)
            [mappings.pop(i) for i in sorted(list(remove_mappings), reverse=True)]

        permuted_events = []
        for m in mappings:
            new_event = deepcopy(self)
            for i in range(len(m)):
                new_event.particles[mappings[0][i]] = deepcopy(self.particles[m[i]]) # Optimisation idea, try change the key (not trivial)
                new_event.checkpoint_particles[mappings[0][i]] = deepcopy(self.checkpoint_particles[m[i]])
            permuted_events.append(new_event)
        return permuted_events
