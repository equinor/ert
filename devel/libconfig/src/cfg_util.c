#include <util.h>
#include <string.h>



/**
    This function shall read a file and tokenize it for proper use with a hierarchical parser.
*/
void cfg_util_create_token_list(const char * file, const char * comment, 
                                const char * sub_start, const char * sub_end,
                                int * num_tokens, char *** tokens)
{
  char * buffer = util_fread_alloc_file_content(file, comment, NULL);

  // Ensure that the sub_start token is separated from any keywords.
  if(sub_start != NULL)
  {
    int len_sub_start = strlen(sub_start);
    char * sub_start_ww = util_malloc( (len_sub_start + 3) * sizeof * sub_start_ww, __func__);
    sub_start_ww[0] = ' ';
    for(int i=0; i<len_sub_start; i++)
    {
      sub_start_ww[1 + i] = sub_start[i];
    }
    sub_start_ww[1+len_sub_start] = ' '; 
    sub_start_ww[2+len_sub_start] = '\0'; 

    char * buffer_tmp = util_string_replace_alloc(buffer, sub_start, sub_start_ww);
    free(buffer);
    free(sub_start_ww);
    buffer = buffer_tmp;
  }

  // Ensure that the sub_end token is separated from any keywords.
  if(sub_end != NULL)
  {
    int len_sub_end = strlen(sub_end);
    char * sub_end_ww = util_malloc( (len_sub_end + 3) * sizeof * sub_end_ww, __func__);
    sub_end_ww[0] = ' ';
    for(int i=0; i<len_sub_end; i++)
    {
      sub_end_ww[1 + i] = sub_end[i];
    }
    sub_end_ww[1+len_sub_end] = ' '; 
    sub_end_ww[2+len_sub_end] = '\0'; 

    char * buffer_tmp = util_string_replace_alloc(buffer, sub_end, sub_end_ww);
    free(buffer);
    free(sub_end_ww);
    buffer = buffer_tmp;
  }

  // Create token list
  util_split_string(buffer, " \t\r\n", num_tokens, tokens);
}



/**
  This function takes a token list and finds the number of tokens in
  the first sequence of tokens such that the number of times sub_start and sub_end 
  has occured is equal. This is used to find the number of tokens belonging 
  to a key in a configuration tree.
*/
int cfg_util_sub_size(int num_tokens, const char ** tokens, const char * sub_start, const char * sub_end)
{
  if(num_tokens == 0)
  {
    return 0;
  }

  bool quit_if_depth_is_zero = false;
  int depth = 0;
  for(int token_nr = 0; token_nr < num_tokens; token_nr++)
  {
    if(strcmp(tokens[token_nr], sub_start) == 0)
    {
      quit_if_depth_is_zero = true;
      depth++;
    }
    if(strcmp(tokens[token_nr], sub_end)   == 0)
    {
      quit_if_depth_is_zero = true;
      depth--;
    }
    if(depth < 0)
    {
      util_abort("%s: Syntax error, too many \"%s\".\n",__func__, sub_end);
    }
    if(depth == 0 && quit_if_depth_is_zero)
    {
      return token_nr + 1;
    }
  }
  if(!quit_if_depth_is_zero && depth == 0)
  {
    return num_tokens;
  }
  else{
    util_abort("%s: Could not match delimiters \"%s\" and \"%s\".\n", __func__, sub_start, sub_end);
    return 0;
  }
}
