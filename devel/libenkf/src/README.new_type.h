/**
The two files README.new_type.c and README.new_type_config.c (along
with the corresponding header files) are meant to serve as a
documentation and reference on how to add new object types to the enkf
system.


new_type.h
==========
When implementing a new type, the header file should contain a typedef like this:

  typedef struct new_type_struct new_type;

This typedef means two things:

  1. The entity "new_type" is to be interpreted as a "struc
     new_type_struct".  

  2. We are informing the compiler that a declaration of the struct
     new_type_struct will come at a later stage. However for the rest
     of the header file we can refer only to "new_type"

The advantages of this way to do it is that only the existence of
"struct new_type_struct" is exported, the actual implementation is
hidden for other files; it is a bit like making all the data private
in a C++ class.

  _____________________ E X A M P L E _____________________________
 /
 | new.h
 | -----
 | typedef struct new_struct new_type;
 | 
 | new_type * new_type_alloc(int , const char *);
 | int        new_get_size(const new_type *);
 | void       new_type_free(new_type *);
 | 
 | 
 | new.c
 | -----
 | #include <util.h>
 | #include <new.h>
 | 
 | struct new_struct {
 |    int    size;
 |    char * name;
 | }
 | 
 | new_type * new_type_alloc(int size, const char * name) {
 |    new_type * new = util_malloc(sizeof * new_type , __func__);
 |    new->size = size;
 |    new->name = util_alloc_string_copy(name);
 |    return new;
 | }
 |
 | int new_get_size(const new_type * new) {
 |    return new->size;
 | }
 | 
 | 
 | void new_type_free(new_type * new) {
 |    free(new->name);
 |    free(new);
 | }
 | 
 | 
 | other.c
 | -------
 | #include <new.h>
 | 
 | void some_func() {
 |    new_type * new = new_type_alloc(100 , "Programmer ...");
 |    ....
 |    ....
 |    new_type_free(new);
 | }
 \_________________________________________________________________

What happen in this little example is the following things:

 1. In the header file "new.h" we say that an implementation of a
    struct new_type_struct will be forthcoming. This struct can
    (without the implementation) be referred to as new_type.

    In the header file we also claim that the three functions:

      i   new_type * new_type_alloc(int , const chat*);
      ii  int        new_get_size(const new_type *);
      iii void       new_type_free(new_type *);

    will be coming.

 2. In the source file new.c we have the implementation of the struct
    new_struct, along with the three functions listed above.

 3. In the third file, other.c which includes "new.h", we can refer to
    the type new_type, and the three functions listed in the
    header. However we can *NOT* get to the fields in the struct of
    type new_type_struct, i.e. code like:
  
        ....
        new->size = 178;
        ....
  
    in "other.c" will *NOT* compile.

*/
