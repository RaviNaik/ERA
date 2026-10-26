# 246_6ffdc8ca7dfac33c.go

341 chars, 16 lines. Open `246_6ffdc8ca7dfac33c.go` beside this file.

## My diagnosis (labels.csv)

- verdict: **fix**
- read: full
- diseases: B3
- evidence:
  - "<reponame>" at offset 0 (line 1, col 1)
  - "<gh_stars>" at offset 26 (line 1, col 27)
- note: <reponame> and <gh_stars> markers on line 1

## Medicine pipeline (medicine.md, B3 > E1 > E2 > E4 > B4)

- action: **fix**
- diseases fired: B3
- agrees with my verdict: **agree**

### Change the medicine makes

```diff
--- before
+++ after
@@ -1,3 +1,2 @@
-<reponame>Grayhaxor/gshark<gh_stars>1-10
 package models
 
```
