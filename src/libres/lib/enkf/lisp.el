(defun insert-cast (type)
  (interactive "sType: ")
  (let ((const (y-or-n-p "Add const modifier")))
    (insert "(")
    (if const (insert "const "))
    (insert (format "%s *) " type))
    (save-buffer)))


(defun insert-guard ()
  (interactive)
  (save-excursion
    (insert "#ifdef __cplusplus\n")
    (insert "extern \"C\" {\n")
    (insert "#endif\n")

    (search-forward "#endif")
    (beginning-of-line)
    (insert "#ifdef __cplusplus\n")
    (insert "}\n")
    (insert "#endif\n"))
  (save-buffer))





