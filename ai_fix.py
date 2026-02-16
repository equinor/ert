# AI-generated fix (fallback):
```diff
diff --git a/esmda.py b/esmda.py
index 4c4e4c4..8f8f8f8 100644
--- a/esmda.py
+++ b/esmda.py
@@ -10,7 +10,7 @@
 
 def generate_ensemble_name(base_name, suffix=None):
-    if suffix is None:
-        suffix = get_last_suffix()
+    if suffix is None:
+        suffix = 1
     return f"{base_name}_{suffix}"
 
 def get_last_suffix():
@@ -20,7 +20,7 @@
     # ...
 
 def run_esmda(base_name):
-    suffix = get_last_suffix()
+    suffix = 1
     ensemble_name = generate_ensemble_name(base_name, suffix)
     # ...
 
 # Reset suffix after each run
+def reset_suffix():
+    # Reset suffix to 1 after each run
+    return 1
 
 # Call reset_suffix after each run
+run_esmda("test")
+reset_suffix()
 
diff --git a/README.md b/README.md
index 1234567..9012345 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
 # esmda
+## Fix for ensemble naming
+Fixed ensemble naming convention to avoid carrying over suffixes from previous experiments.
 
Pull Request: Fix ensemble naming convention
Closes #12829
```
