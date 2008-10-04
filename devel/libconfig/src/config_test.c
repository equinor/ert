#include <stdlib.h>
#include <stringlist.h>
#include <hash.h>
#include <config.h>


int main(void) {
  const char * config_file = "config_test_input";
  config_type * config = config_alloc();
  config_item_type * item;
  
  item = config_add_item(config , "KEY1" , true  , true);
  item = config_add_item(config , "KEY2" , true  , false);
  config_item_set_argc_minmax(item , 1 , 4 , (const config_item_types [4]) {CONFIG_EXECUTABLE , CONFIG_EXISTING_FILE , CONFIG_BOOLEAN , CONFIG_BOOLEAN});


  item = config_add_item(config , "FATHER"  , false , false);
  {
    stringlist_type * children = stringlist_alloc_argv_ref( (const char *[2]) {"CHILD1" , "CHILD2"} , 2);
    config_item_set_required_children(item , children);
    stringlist_free(children);
  }
  item = config_add_item(config , "CHILD1"  , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
  
  config_parse(config , config_file , "--" , "INCLUDE" , true , true);
  


  {
    stringlist_type * sl = config_alloc_complete_stringlist(config , "KEY1");
    char * s = stringlist_alloc_joined_string(sl , "|");
    printf("KEY1 -> \"%s\" \n",s);
    printf("CONFIG_IGET:%s\n" , config_iget(config , "KEY2" , 0));
    free(s);
    stringlist_free(sl);
  }
  
  
  config_free(config);
}
