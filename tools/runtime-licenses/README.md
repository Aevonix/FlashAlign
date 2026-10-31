# Runtime archive license templates

The native bundle builder copies these templates into the runtime archive.
`NOTICE.md` and `COMMERCIAL.md` go at its root; `FLASHALIGN-LICENSE.md` goes
under `sources/`. Their relative links target that archive layout and resolve
after packaging. The archive validator checks the completed layout.

The exact TensorFold source snapshot and its original notices are included
separately. These templates describe the FlashAlign additions without replacing
upstream or dependency licenses.
