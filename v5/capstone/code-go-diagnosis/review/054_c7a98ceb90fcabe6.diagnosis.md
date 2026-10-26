# 054_c7a98ceb90fcabe6.go

1371 chars, 60 lines. Open `054_c7a98ceb90fcabe6.go` beside this file.

## My diagnosis (labels.csv)

- verdict: **fix**
- read: full
- diseases: B3
- evidence:
  - "<reponame>" at offset 0 (line 1, col 1)
  - "<filename>" at offset 36 (line 1, col 37)
- note: two joined markers on line 1

## Medicine pipeline (medicine.md, B3 > E1 > E2 > E4 > B4)

- action: **fix**
- diseases fired: B3
- agrees with my verdict: **agree**

### Change the medicine makes

```diff
--- before
+++ after
@@ -1,3 +1,2 @@
-<reponame>gsasha/hvac_ip_mqtt_bridge<filename>hvac/models/samsung/samsung.go
 package samsung
 
```
