#include <cfg_util.h>
#include <cfg_lang.h>
#include <string.h>

int main()
{
  /*
  int num_tokens = 0;
  char ** tokens;
  cfg_util_create_token_list("svada.txt", "--", "{", "}", &num_tokens, &tokens);

  printf("num_tokens : %i\n", num_tokens);

  for(int i=0; i<num_tokens; i++)
  {
    printf("token %i : %s\n", i, tokens[i]);
    if(strcmp(tokens[i], "{") == 0)
    {
      int key_size = cfg_util_key_size(num_tokens-i, tokens+i, "{", "}");
      printf("key_size: %i\n", key_size);
    }
  }
  */

  /*

  char * tokens[] = {"restriction","kakk","madda","fakka","data_type","string","help_text","rtfm!"};

  cfg_key_def_type * cfg_key_def = cfg_key_def_alloc_from_tokens(8, tokens, "svada");

  cfg_key_def_printf(cfg_key_def);


  char * tokens2[] = {"reinspikka hiphop","{","foo","bar","}"};
  printf("%i\n", cfg_util_sub_size(5, tokens2, "{","}"));


  char * tokens3[] = {"help_text","read the fucking manual","required","key","svada","{","help_text","rtfm","data_type","string","}"};
  cfg_type_def_type * cfg_type_def = cfg_type_def_alloc_from_tokens(11, tokens3, "foobar");
  cfg_type_def_printf(cfg_type_def);
  
  */

  char * pad_keys[] = { "{" ,"}" ,"=" };
  char * buffer = cfg_util_alloc_token_buffer("svada.txt", "--", 3, pad_keys);
  printf("%s\n",buffer);

  char * buffer_wrk = buffer;
  char * token = cfg_util_alloc_next_token(&buffer_wrk);
  while(token != NULL)
  {
    printf("token : %s\n", token);
    free(token);
    token = cfg_util_alloc_next_token(&buffer_wrk);
  }

  printf("buffer : %s\n", buffer);
  free(buffer);

  return 0;
}
