# AI-generated fix (fallback):
```diff
PR: Rename connect function to establish_server_connection for clarity
Closes #12882

diff --git a/file.py b/file.py
index 1234567..8901234 100644
--- a/file.py
+++ b/file.py
@@ -1,7 +1,7 @@
 def establish_server_connection(
-    def connect(
     *,
     project: os.PathLike[str],
     timeout: int | None = None,
     logging_config: str | None = None,
-) -> ErtServer:
+) -> ErtServer:
```
