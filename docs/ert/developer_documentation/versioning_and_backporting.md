Versioning and Backporting
==========================

Ert generally has a new major or minor version every month, which we will refer
to as the current 'stable version'. In between, there are usually bug-fix
releases and there can be development releases. In general, which changes are
applied to the current stable release is somewhat conservative, limited to
fixing bugs or applying extra telemetry.


# The process

Following the internal release process timeline (komodo), a new major or minor
version tag is created using the github releases interface roughly every month.
This creates either [a new major or minor version](#is-it-a-minor-version-or-a-major-version-increment).
This then auto generates release notes which are based on the PR titles
(see [the pr template](../../../.github/PULL_REQUEST_TEMPLATE.md) for mor information
about tailoring PR titles and labels for the release notes).


# Backporting

After the release of a new stable version, bug-fixes are backported to it. In
order to do so, a version branch is created for the first backport. If the
new stable version is "ert-22.1.0", the name of the version branch should be
"version-22.1". We also create a new label for prs called "backport version-22.1".
When a PR to main is marked with this label, a workflow will trigger once it is
merged that cherry-picks the change to the "version-22.1" branch.

The default procedure is to first create a bug-fix PR towards main then
automatically create a PR towards the version branch. This avoids having
changes to version branches that are not also in main, which can create
confusion as to what has been released to users.


# Is it a minor version or major version increment?

We follow [semantic versioning](https://semver.org/). Here is a list of common
scenarios, and whether they trigger a major version, or minor version increment:

1. It is a breaking change to change the plugin API
2. It is a breaking change to apply a non-reversible storage migration
3. It is a breaking change to require a newer version of public dependencies
   (Go from supporting both numpy 1 and numpy 2 to requiring numpy>=2).
4. It is a breaking change to require a newer version of plugins (Require newer
   versions/changes to semeio, everest-models, subsurface etc.).
5. New features, ie. minor version increment, is any behavior a user or plugin may
   come to depend upon: additions to plugin api, new keywords, new options in the
   GUI etc.
