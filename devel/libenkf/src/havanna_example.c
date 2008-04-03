#include <util.h>
#include <hash.h>
#include <void_arg.h>





int main(int argc , char ** argv) {
  /*
    These are the keys we will search for in the template file.
  */
  const char * int_key       = "FLAG";
  const char * double_key    = "VALUE";
  const char * path_key      = "OUTPATH";
  const char * template_file = "template";

  /*
    The keys to replace, along with their value, are stored in 
    a hash table.
  */
  hash_type * kw_hash   = hash_alloc();

  
  /*
    The values we are substituting in are stored in an object called
    void_arg_type (the implementation of this object is part of
    libutil). The point about this object is that it stores anonymous
    data along with a type identifier, saying which type
    (i.e. int/double/string/ ...) a certain amount of data represents.

    We start by allocating the void_arg instances, but without sending
    in a value.
  */
    
  void_arg_type * flag  = void_arg_alloc_int(0);  
  void_arg_type * value = void_arg_alloc_double(0);
  void_arg_type * path  = void_arg_alloc_ptr(NULL);

  
  /*
    We insert all the void_arg instances (as placeholders for the
    actual data) in the kw_hash hash_table. Observe that we use the
    xxx_hash_owned_ref() function to insert the void_arg instances,
    and pass a destructor, this way the hash code will clear up
    void_arg instances when it is shut down.
  */

  hash_insert_hash_owned_ref(kw_hash , int_key    , flag  , void_arg_free__);
  hash_insert_hash_owned_ref(kw_hash , double_key , value , void_arg_free__);
  hash_insert_hash_owned_ref(kw_hash , path_key   , path  , void_arg_free__);
  hash_insert_ref(kw_hash , "extra" , NULL);
  
  while (1) {
    int    flag_value;
    double value_value;
    char path_value[128];
    char target_file[128];

    printf("Give a value for flag  (integer)...: "); fscanf(stdin , "%d"  , &flag_value) ;  
    printf("Give a value for value (double)....: "); fscanf(stdin , "%lg" , &value_value); 
    printf("Give a value for path (string) ....: "); fscanf(stdin , "%s"  ,  path_value) ; 
    printf("Where do you want to store this ...: "); fscanf(stdin , "%s"  ,  target_file); 

    /*
      We insert values in the void_arg containers.
    */
    void_arg_pack_int(flag     , 0 , flag_value );
    void_arg_pack_double(value , 0 , value_value);
    void_arg_pack_ptr(path     , 0 , path_value );


    /* 
       We call the filter file to actually do the replacement. We have
       asked the filter routine for all warnings.
    */
    util_filter_file(template_file , NULL , target_file , '<' , '>' , kw_hash , util_filter_warn_unknown);
    printf("Have written file:%s with substituted values.\n",target_file);
    printf("-----------------------------------------------------------------\n");
    break;
  }
  
  hash_free(kw_hash);
}
