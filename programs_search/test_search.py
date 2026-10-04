"""Small mathematical and integration checks without saved certificate data."""
from fractions import Fraction as F
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .certificate_format import write_transport, verify_transport_polynomial, multiply
from .generate import main, validate_settings
from .primal_model import Primal
from .search_reference import cubic_positive


class SearchTests(unittest.TestCase):
    def test_positive_cubic_identity(self):
        a, b = np.meshgrid(np.arange(5, dtype=float), np.arange(5, dtype=float))
        x = cubic_positive(a, b)
        residual = (x*x-a)*x-b
        scale = 1 + abs(x)**3 + abs(a*x) + abs(b)
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(abs(residual) <= 16*np.finfo(float).eps*scale))

    def test_exact_polynomial_serialization(self):
        # A synthetic potential q(2-q)^2, unrelated to any computed certificate.
        polynomial = multiply([0, 1], [4, -4, 1])
        source = {'scales': [1, 2], 'labels': ['xi_0', 'xi_1', 'tau', 'chi', 'u0:0'],
                  'parameters': [0, 0, 0, 0, 1],
                  'basis': [None, None, None, None, {'q_polynomial': [str(x) for x in polynomial]}]}
        with tempfile.TemporaryDirectory() as td:
            proposal, output = Path(td)/'search.json', Path(td)/'transport.txt'
            proposal.write_text(json.dumps(source))
            write_transport(proposal, output)
            decoded = verify_transport_polynomial(output)
            self.assertEqual([F(x) for x in decoded['terms'][0]['polynomial']], polynomial)
            tokens = output.read_text().split()
            # Changing a stored quotient must violate the exact cancellation.
            tokens[-1] = str(F(tokens[-1]) + 1)
            output.write_text(' '.join(tokens))
            with self.assertRaises(ValueError):
                verify_transport_polynomial(output)

    def test_primal_variational_derivatives(self):
        # A regular synthetic polynomial, with no fitted input coefficients.
        model = Primal([1, 2, 4], degree=6, order=32, second=True)
        coefficients = np.zeros(7)
        coefficients[0] = np.log(2.)
        value = model.evaluate(coefficients)
        step = np.finfo(float).eps**(1/3)
        for j in (0, 3, 6):
            direction = np.eye(7)[j] * step
            plus, minus = model.evaluate(coefficients+direction), model.evaluate(coefficients-direction)
            derivative = (plus['observations']-minus['observations'])/(2*step)
            np.testing.assert_allclose(derivative, value['jacobian'][:,j], rtol=step, atol=step)
            energy_derivative = (plus['energy']-minus['energy'])/(2*step)
            self.assertLess(abs(energy_derivative-value['energy_gradient'][j]), step)

    def test_nonempty_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'inputs'; output.mkdir()
            marker = output/'keep.txt'; marker.write_text('keep')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--output', str(output), '--work', str(Path(td)/'work')])
            self.assertEqual(marker.read_text(), 'keep')
            self.assertFalse((Path(td)/'work').exists())

    def test_settings_have_no_candidate_parameters(self):
        data = json.loads(Path(__file__).with_name('settings.json').read_text())
        validate_settings(data)
        data['beta'] = 1
        with self.assertRaises(ValueError):
            validate_settings(data)


if __name__ == '__main__':
    unittest.main()
