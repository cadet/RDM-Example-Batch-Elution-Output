# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# (batch_elution)=
# # Batch Elution
# tags: batch elution, langmuir, optimization
#
# Batch elution is the most basic mode of preparative chromatography.
# A pulse of feed is injected onto the column and then eluted with eluent.
# The components travel through the column at different velocities and leave it as separate peaks, which are collected as fractions.
#
# ```{figure} ./figures/flow_sheet.svg
# Flow sheet for batch elution process.
# ```
#
# The first part sets up a batch elution process for a two-component system with a Langmuir isotherm.
# The following parts optimize the operating conditions of this process, first for a single objective and then for multiple objectives.
#
# ```{toctree}
# :maxdepth: 1
#
# process
# optimization_single
# optimization_multi
# ```
