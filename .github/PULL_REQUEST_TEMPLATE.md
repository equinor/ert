**Issue**
Resolves #my_issue


**Approach**
_Short description of the approach_

(Screenshot of new behavior in GUI if applicable)


- [ ] PR title captures the intent of the changes, and is fitting for release notes.
- [ ] Added appropriate release note label
- [ ] Commit history is consistent and clean, in line with the [contribution guidelines](https://github.com/equinor/ert/blob/main/CONTRIBUTING.md).
- [ ] Make sure unit tests pass locally after every commit (`git rebase -i main
      --exec 'just rapid-tests'`)

## When applicable
- [ ] **When there are user facing changes**: Updated documentation
- [ ] **New behavior or changes to existing untested code**: Ensured that unit tests are added (See [Ground Rules](https://github.com/equinor/ert/blob/main/CONTRIBUTING.md#ground-rules)).
- [ ] **Large PR**: Prepare changes in small commits for more convenient review
- [ ] **Bug fix**: Add regression test for the bug
- [ ] **Bug fix**: Create Backport PR to latest release

<!--
Adding labels helps the maintainers when writing release notes. This is the [list of release note labels](https://github.com/equinor/ert/labels?q=release-notes).
-->
