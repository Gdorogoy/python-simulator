"""
Guidance reads the estimated state (from navigation, never the true one) plus the
target, and computes the desired attitude/thrust command. Navigation's estimate is
read-only input here; the real state only changes in dynamics, at the end of the tick.
"""