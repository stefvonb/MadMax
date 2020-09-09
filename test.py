#!/usr/bin/env python3

import unittest
from madmax import change_of_variables as cv
from madmax.objects import FourMomentum, Event, Particle
import madmax.nlopt_interface as nl

class TestClasses(unittest.TestCase):

    def test_energy_scaling(self):
        test_particle = Particle(0, 0, FourMomentum(20.0, 5.0, 5.0, 5.0))
        prescaled_energy = test_particle.E
        prescaled_mass = test_particle.mass()
        test_particle.scale_energy(2.0)
        scaled_energy = test_particle.E
        scaled_mass = test_particle.mass()
        self.assertEqual(scaled_energy, 2.0*prescaled_energy)
        self.assertAlmostEqual(prescaled_mass, scaled_mass, 5)

    def test_permutations(self):
        test_event = Event(13000)
        test_event.x1 = 0.5
        test_event.x2 = 0.1
        test_event.add_particle("j/1", Particle(2, 2, FourMomentum(10.0, 1.0, 1.0, 1.0)))
        test_event.add_particle("j/2", Particle(-3, 2, FourMomentum(20.0, 2.0, 2.0, 2.0)))
        test_event.add_particle("v1", Particle(0, 0, FourMomentum()))
        test_event.add_particle("v2", Particle(0, 0, FourMomentum()))
        test_event.add_particle("l/1", Particle(-11, 2, FourMomentum(100.0, 10.0, 10.0, 10.0)))
        test_event.add_particle("l/2", Particle(11, 2, FourMomentum(200.0, 20.0, 20.0, 20.0)))
        test_event.add_particle("l/3", Particle(13, 2, FourMomentum(300.0, 30.0, 30.0, 30.0)))
        test_event.add_particle("b/1", Particle(5, 2, FourMomentum(1000.0, 100.0, 100.0, 100.0)))
        permutations = test_event.make_permutations()
        # Should add some extra code to check the permutations, but for now the inspection seems okay
        self.assertEqual(len(permutations), 2*1*3*2*1*1)

    def test_class_B(self):
        """
        Testing with W+y LHE event:
        -1 -1  0  0    0  501 -0.0000000000e+00 +0.0000000000e+00 +7.4827774544e+00 7.4827774544e+00
        2  -1  0  0  501    0 +0.0000000000e+00 -0.0000000000e+00 -4.0157473806e+02 4.0157473806e+02
        24  2  1  2    0    0 +1.4717751207e+01 +1.9702783714e+01 -2.5724660044e+02 2.7001987857e+02
        -13 1  3  3    0    0 +3.2395050227e+01 -5.3817378176e+00 -4.5790709849e+01 5.6348837518e+01
        14  1  3  3    0    0 -1.7677299020e+01 +2.5084521531e+01 -2.1145589059e+02 2.1367104106e+02
        22  1  1  2    0    0 -1.4717751207e+01 -1.9702783714e+01 -1.3684536016e+02 1.3903763694e+02
        """
        p4_w = FourMomentum(2.7001987857e+02, 1.4717751207e+01, 1.9702783714e+01, -2.5724660044e+02)
        p4_l = FourMomentum(5.6348837518e+01, 3.2395050227e+01, -5.3817378176e+00, -4.5790709849e+01)
        p4_nu = FourMomentum(2.1367104106e+02, -1.7677299020e+01, +2.5084521531e+01, -2.1145589059e+02)
        p4_y = FourMomentum(1.3903763694e+02, -1.4717751207e+01, -1.9702783714e+01, -1.3684536016e+02)
        x1 = 7.4827774544e+00/6500.0
        x2 = 4.0157473806e+02/6500.0

        input_event = Event(13000, l=Particle(-13, 2, p4_l), y=Particle(22, 2, p4_y))

        class_B_sols = cv.class_B(input_event, ["nu", "l"], p4_w.mass())

        passed = False
        for sol in class_B_sols:
            if sol["nu"].four_momentum == p4_nu and abs(sol.x1 - x1) < 1E-5 and abs(sol.x2 - x2) < 1E-5:
                passed = True

        self.assertTrue(passed)

    def test_class_D(self):
        """
        Testing with LHE event:
         21 -1  0  0 501 502 +0.00000000000e+00 +0.00000000000e+00 +1.54270123210e+02  1.54270123210e+02
         21 -1  0  0 502 503 +0.00000000000e+00 +0.00000000000e+00 -1.78313397110e+03  1.78313397110e+03
         6   2  1  2 501   0 +2.21796373830e+02 -1.39027387052e+02 -6.28762112083e+02  7.02379511610e+02
         5   1  3  3 501   0 +8.09346302052e+01 -1.07816045715e+02 -3.87091333836e+02  4.09922559527e+02
         24  2  3  3   0   0 +1.40861743625e+02 -3.12113413375e+01 -2.41670778248e+02  2.92456952084e+02
        -13  1  5  5   0   0 +1.19984134623e+02 -4.38501699278e+01 -2.34469954878e+02  2.67011590955e+02
         14  1  5  5   0   0 +2.08776090020e+01 +1.26388285904e+01 -7.20082336998e+00  2.54453611289e+01
        -6   2  1  2   0 503 -1.39127363300e+02 +1.48759661306e+02 -1.04441511879e+03  1.07844458284e+03
        -5   1  8  8   0 503 -1.23288432443e+02 +2.98887004115e+01 -5.95113687202e+02  6.08502886337e+02
        -24  2  8  8   0   0 -1.58389308569e+01 +1.18870960894e+02 -4.49301431586e+02  4.69941696503e+02
         13  1 10 10   0   0 -3.56865149184e+01 +4.85974772445e+01 -2.63406562920e+02  2.70218910387e+02
        -14  1 10 10   0   0 +1.98475840615e+01 +7.02734836498e+01 -1.85894868666e+02  1.99722786115e+02
         25  2  1  2   0   0 -8.26690105298e+01 -9.73227425370e+00 +4.43133829811e+01  1.56579999860e+02
         5   1 13 13 504   0 -1.10535285330e+02 +1.52021136715e+01 +4.01076612446e+01  1.18658619796e+02
        -5   1 13 13   0 504 +2.78662748006e+01 -2.49343879252e+01 +4.20572173656e+00  3.79213800645e+01
        """
        p4_t1 = FourMomentum(7.02379511610e+02, +2.21796373830e+02, -1.39027387052e+02, -6.28762112083e+02)
        p4_t2 = FourMomentum(1.07844458284e+03, -1.39127363300e+02, +1.48759661306e+02, -1.04441511879e+03)
        p4_W1 = FourMomentum(2.92456952084e+02, +1.40861743625e+02, -3.12113413375e+01, -2.41670778248e+02)
        p4_W2 = FourMomentum(4.69941696503e+02, -1.58389308569e+01, +1.18870960894e+02, -4.49301431586e+02)
        p4_b1 = FourMomentum(4.09922559527e+02, +8.09346302052e+01, -1.07816045715e+02, -3.87091333836e+02)
        p4_b2 = FourMomentum(6.08502886337e+02, -1.23288432443e+02, +2.98887004115e+01, -5.95113687202e+02)
        p4_mu1 = FourMomentum(2.67011590955e+02, +1.19984134623e+02, -4.38501699278e+01, -2.34469954878e+02)
        p4_mu2 = FourMomentum(2.70218910387e+02, -3.56865149184e+01, +4.85974772445e+01, -2.63406562920e+02)
        p4_nu1 = FourMomentum(2.54453611289e+01, +2.08776090020e+01, +1.26388285904e+01, -7.20082336998e+00)
        p4_nu2 = FourMomentum(1.99722786115e+02, +1.98475840615e+01, +7.02734836498e+01, -1.85894868666e+02)
        p4_h = FourMomentum(1.56579999860e+02, -8.26690105298e+01, -9.73227425370e+00, +4.43133829811e+01)
        p4_hb1 = FourMomentum(1.18658619796e+02, -1.10535285330e+02, +1.52021136715e+01, +4.01076612446e+01)
        p4_hb2 = FourMomentum(3.79213800645e+01, +2.78662748006e+01, -2.49343879252e+01, +4.20572173656e+00)
        x1 = 1.54270123210e+02/6500.0
        x2 = 1.78313397110e+03/6500.0

        input_event = Event(13000, mu1=Particle(-13, 2, p4_mu1), b1=Particle(5, 2, p4_b1), 
            mu2=Particle(13, 2, p4_mu2), b2=Particle(-5, 2, p4_b2), hb1=Particle(5, 2, p4_hb1), hb2=Particle(-5, 2, p4_hb2))

        class_D_sols = cv.class_D(input_event, ["nu1", "nu2", "mu1", "b1", "mu2", "b2"], p4_t1.mass(), p4_t2.mass(), p4_W1.mass(), p4_W2.mass())
        passed = False
        for sol in class_D_sols:
            if sol["nu1"].four_momentum == p4_nu1 and sol["nu2"].four_momentum == p4_nu2 and abs(sol.x1 - x1) < 1E-5 and abs(sol.x2 - x2) < 1E-5:
                passed = True

        self.assertTrue(passed)

    def test_class_E(self):
        """
        Testing with LHE event:
         21 -1 0 0 502 501 +0.0000000000e+00 +0.0000000000e+00 +4.1758290104e+02 4.1758290104e+02
         21 -1 0 0 501 502 -0.0000000000e+00 -0.0000000000e+00 -9.3527493892e+00 9.3527493892e+00
         25  2 1 2   0   0 +0.0000000000e+00 -1.7763568394e-15 +4.0823015165e+02 4.2693565043e+02
        -24  2 3 3   0   0 +3.1392005803e+01 +3.9830546942e+00 +2.4197449191e+02 2.5697832303e+02
        -11  1 3 3   0   0 -1.8065699537e+01 +5.1144469720e+00 +7.7093957249e+01 7.9347371172e+01
         13  1 4 4   0   0 +3.6021334597e+01 +3.3915572129e+01 +1.9426779434e+02 2.0046889658e+02
         12  1 3 3   0   0 -1.3326306266e+01 -9.0975016661e+00 +8.9161702490e+01 9.0609956220e+01
        -14  1 4 4   0   0 -4.6293287943e+00 -2.9932517435e+01 +4.7706697567e+01 5.6509426451e+01
        """
        p4_e = FourMomentum(7.9347371172e+01, -1.8065699537e+01, +5.1144469720e+00, +7.7093957249e+01)
        p4_mu = FourMomentum(2.0046889658e+02, +3.6021334597e+01, +3.3915572129e+01, +1.9426779434e+02)
        p4_ve = FourMomentum(9.0609956220e+01, -1.3326306266e+01, -9.0975016661e+00, +8.9161702490e+01)
        p4_vm = FourMomentum(5.6509426451e+01, -4.6293287943e+00, -2.9932517435e+01, +4.7706697567e+01)

        p4_w1 = p4_e + p4_ve
        p4_w2 = p4_mu + p4_vm
        p4_h = p4_w1 + p4_w2

        input_event = Event(13000)
        input_event.add_particle("e-", Particle(-11, 2, p4_e))
        input_event.add_particle("mu+", Particle(13, 2, p4_mu))

        class_E_sols = cv.class_E(input_event, ["nu1", "nu2", "e-", "mu+"], p4_w1.mass(), p4_w2.mass(), p4_h.mass(), 1.89940637)

        passed = False
        for sol in class_E_sols:
            if p4_ve == sol["nu1"].four_momentum and p4_vm == sol["nu2"].four_momentum:
                passed = True
        self.assertTrue(passed)

    def test_block_D(self):
        """
        Testing with a Higgs -> bb event
         25  2  1  2   0   0 -8.26690105298e+01 -9.73227425370e+00 +4.43133829811e+01  1.56579999860e+02
         5   1 13 13 504   0 -1.10535285330e+02 +1.52021136715e+01 +4.01076612446e+01  1.18658619796e+02
        -5   1 13 13   0 504 +2.78662748006e+01 -2.49343879252e+01 +4.20572173656e+00  3.79213800645e+01
        """

        p4_b1 = FourMomentum(1.18658619796e+02, -1.10535285330e+02, +1.52021136715e+01, +4.01076612446e+01)
        p4_b2 = FourMomentum(3.79213800645e+01, +2.78662748006e+01, -2.49343879252e+01, +4.20572173656e+00)
        p4_h = FourMomentum(1.56579999860e+02, -8.26690105298e+01, -9.73227425370e+00, +4.43133829811e+01)

        input_event = Event(13000, pb1=Particle(5, 2, p4_b1), pb2=Particle(5, 2, p4_b2))
        block_D_sols = cv.block_D(input_event, ["pb1", "pb2"], p4_h.mass(), p4_b1.theta, p4_b1.phi, p4_b1.mass())
        
        passed = False
        for sol in block_D_sols:
            if p4_b1 == sol["pb1"].four_momentum:
                passed = True
        self.assertTrue(passed)

#    def test_nlopt(self):
#        for n, algorithm in nl.NLOPT_ALGORITHMS.items():
            # Initialise from integer
#            opt_i = nl.optimiser(n, 4)
#            self.assertTrue(opt_i.get_algorithm_name() == algorithm['desc'])
            # Initialise from string
#            opt_s = nl.optimiser(algorithm['name'], 4)
#            self.assertTrue(opt_s.get_algorithm() == n)
        
        # Test for non-existant algorithms
#        self.assertRaises(nl.NLOPTError, nl.optimiser, 45, 4)
#        self.assertRaises(nl.NLOPTError, nl.optimiser, "Somebody once told me the world is gonna roll me", 4)
#        self.assertRaises(nl.NLOPTError, nl.optimiser, 2.5, 4)
#        self.assertRaises(nl.NLOPTError, nl.optimiser, 2, 4.5)

if __name__ == "__main__":
    unittest.main()
