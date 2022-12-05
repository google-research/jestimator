# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/jestimator/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.3.3] - 2022-12-01

* Python 3.7 compatibility.
* Add `d_coef` and `c_coef` to Amos hyper-parameter.
* Support for flax_mutables in checkpointing.
* Bug fix data/pipeline_rec and simplified data_utils code.

## [0.3.2] - 2022-11-01

* The Amos optimizer implementation stick to the paper.
* Initial Pypi package.
* Setup Github workflow for unit test and auto-publish.
* MNIST examples.

[Unreleased]: https://github.com/google-research/jestimator/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/google-research/jestimator/releases/tag/v0.3.3
[0.3.2]: https://github.com/google-research/jestimator/releases/tag/v0.3.2
