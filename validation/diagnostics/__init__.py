"""Identification diagnostics for Frugal Flows (scripts, not notebooks).

This package reproduces the *identification* findings of
``validation/Continous_Frugal_Flows.ipynb`` as runnable, sense-checkable
scripts. It is purely additive: it only imports and calls existing code in
``frugal_flows/`` and ``validation/`` — it never modifies it.

Run everything inside the one working env:

    micromamba run -n frugal-flows-flowjax python -m validation.diagnostics.d1_ate_recovery --help

(or ``cd validation && python -m diagnostics.d1_ate_recovery`` — see ``_harness``
for the path handling that makes the existing ``data_processing_and_simulations``
package importable from here.)
"""
