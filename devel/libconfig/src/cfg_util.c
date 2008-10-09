#include <assert.h>
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



/*
  This function creates a string buffer from a file. Furthermore, if the strings in pad_keys are found in the buffer,
  they are padded with a space before and after.

  I.e., if the file contains

  key=value

  and "=" is in pad_keys, then the buffer will read

  key = value


*/
char * cfg_util_alloc_token_buffer(const char * file, const char * comment, int num_pad_keys, const char ** pad_keys)
{
  char * buffer_wrk = util_fread_alloc_file_content(file, comment, NULL);
  

  char ** padded_keys = util_malloc(num_pad_keys * sizeof * padded_keys, __func__);
  for(int key_nr = 0; key_nr < num_pad_keys; key_nr++)
  {
    assert(pad_keys[key_nr] != NULL);

    int key_len = strlen(pad_keys[key_nr]);
    padded_keys[key_nr] = util_malloc( (key_len + 3) * sizeof * padded_keys[key_nr], __func__);
    padded_keys[key_nr][0] = ' ';
    for(int i=0; i<key_len; i++)
    {
      padded_keys[key_nr][1+i] = pad_keys[key_nr][i];
    }
    padded_keys[key_nr][key_len + 1] = ' ';
    padded_keys[key_nr][key_len + 2] = '\0';
    printf("padded_key %i : %s\n", key_nr, padded_keys[key_nr]);
  }
  char * buffer = util_string_replacen_alloc(buffer_wrk, num_pad_keys, pad_keys, (const char **) padded_keys);
  free(buffer_wrk);
  util_free_stringlist(padded_keys, num_pad_keys);
  return buffer;
}



/*
  This function takes a pointer to a position in a string and returns a copy
  of the next token in the string. Furthermore, the position in the string is
  moved to the position after the token.
*/
char * cfg_util_alloc_next_token(char ** buff_pos)
{
  char * sep = " \t\r\n";
  int init_whitespace = strspn(*buff_pos, sep);
  if(init_whitespace > 0)
    *buff_pos += init_whitespace;

  bool quoted;
  int len_token;
  if(*buff_pos[0] == '"')
  {
    quoted = true;
    *buff_pos += 1;
    len_token = strcspn(*buff_pos, "\"");
  }
  else if(*buff_pos[0] == '\'')
  {
    quoted = true;
    *buff_pos += 1;
    len_token = strcspn(*buff_pos, "\'");
  }
  else
  {
    quoted = false;
    len_token = strcspn(*buff_pos, sep);
  }

  if(len_token == 0)
    return NULL;

  char * token = util_malloc( (len_token + 1) * sizeof * token, __func__);
  memmove(token, *buff_pos, len_token);
  token[len_token] = '\0';
  *buff_pos += len_token;
  
  if(quoted)
    *buff_pos += 1;

  return token;
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
