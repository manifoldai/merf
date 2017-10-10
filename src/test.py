"""
Simple Test File

:copyright: 2017 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import sys
sys.path.append('..')

from src.utils import generate_test_data
from src.merf import Merf

y, X, Z, clusters, b, ptev, prev = generate_test_data([90, 90, 50, 50], m=1, sigma_b=2, sigma_e=1)
mrf = Merf(n_estimators=100, max_iterations=10)
mrf.fit(X, Z, clusters, y)
