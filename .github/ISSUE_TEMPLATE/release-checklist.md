---
name: Release checklist
about: This is the checklist to go through before each new release
title: Release checklist for v
labels: ''
assignees: ''

---

# Checklist
- [ ] Bump software version
- [ ] Write the changelog for the new release
- [ ] Assign relevant issues and PRs to the corresponding milestone
- [ ] Regenerate the C files from the Cython sources (if they are changed)
- [ ] Update the submodule for the backend (if it is changed)
- [ ] Mirror both the frontend and the backend to the NSLS-II internal Gitlab  
- [ ] Update NSLS-II Conda recipe and build the Conda package
