# How to make a new Bempp-cl release

## Updating the version number
You will need to update the version number in three places. These are:

- [VERSION](VERSION)
- [bempp_cl/version.py](bempp_cl/version.py)
- [codemeta.json](codemeta.json)
- [pyproject.toml](pyporoject.toml)

We use the format `{a}.{b}.{c}` for the version numbers, where
`{a}` should be increased for major releases,
`{b}` should be increased for larger changes, and
`{c}` should be increased for bugfixes and other minor changes.

## Making the release on GitHub
Once the version numbers are updated on the `main` branch, you need to
[create a new release on GitHub](https://github.com/bempp/bempp-cl/releases).
The release should be tagged `v{a}.{b}.{c}` with `{a}`, `{b}`, and `{c}` replaced
with the three parts of the version number.

You should include a bullet pointed list of the main changes since the last version in
the "Describe this release" section.

## Updating bempp.com
Open a pull request to the [bempp-website repo](https://github.com/bempp/bempp-website) with the following changes:

- Update `bemppversion` in the file `_config.yml`
- Add the new version and release notes to `changelog.md`

## Pushing to PyPI
Once a version is created on GitHub, the new version should automatically be pushed to
PyPI by [GitHub Actions](https://github.com/bempp/bempp-cl/actions/workflows/release.yml).
If this doesn't work, ask @mscroggs to look into why it's broken.

## Updating the Conda package
A short time after the PyPI package is updated, a conda bot will open
[a pull request into the Bempp-cl conda feedstock repo](https://github.com/conda-forge/bempp-cl-feedstock/pulls).
Once this is merged, the new version will be available via conda.
