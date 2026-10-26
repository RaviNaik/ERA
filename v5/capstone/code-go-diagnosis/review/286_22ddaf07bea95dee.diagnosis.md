# 286_22ddaf07bea95dee.go

954 chars, 48 lines. Open `286_22ddaf07bea95dee.go` beside this file.

## My diagnosis (labels.csv)

- verdict: **fix**
- read: full
- diseases: B3
- evidence:
  - "<gh_stars>" at offset 0 (line 1, col 1)
- note: <gh_stars>0 marker on line 1

## Medicine pipeline (medicine.md, B3 > E1 > E2 > E4 > B4)

- action: **fix**
- diseases fired: B3
- agrees with my verdict: **agree**

### Change the medicine makes

```diff
--- before
+++ after
@@ -1,3 +1,2 @@
-<gh_stars>0
 //+build darwin,!arm,!arm64
 
```
