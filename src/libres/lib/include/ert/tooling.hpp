#ifndef RES_TOOLING_HPP
#define RES_TOOLING_HPP

// The following defines are used to mark declarations as potentially used
// this to avoid warnings from the compiler that a declaration is not used.

// This define is used for seemingly unused declarations but usually are indirectly
// used in C code. For example through a macro expansion.
#define C_USED [[gnu::used]]

// This define is used for declarations that are only used in Python.
#define PY_USED [[gnu::used]]

#endif
