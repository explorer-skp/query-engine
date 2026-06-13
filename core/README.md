# core/

Value vocabulary and the typed column batch — `Type`, `Column`, `SelectionVector`,
`Batch`, `Schema`. The spine every operator and kernel reads and writes.

**WP-0 status:** DRAFT interface stubs only (`types.h`, `column.h`). Frozen at WP-1.
No `.cpp` logic yet. From-scratch: no query-engine/dataframe libraries here.
