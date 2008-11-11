#include <assert.h>
#include <util.h>
#include <string.h>
#include <conf_util.h>



/*
  This function creates a string buffer from a file. Furthermore, if the strings in pad_keys are found in the buffer,
  they are padded with a space before and after.

  I.e., if the file contains

  key=value

  and "=" is in pad_keys, then the buffer will read

  key = value
*/
static
char * __conf_util_fscanf_alloc_token_buffer(
  const char * file,
  const char * comment,
  int          num_pad_keys,
  const char ** pad_keys)
{
  char * buffer_wrk = util_fread_alloc_file_content(file, comment, NULL);
  

  char ** padded_keys = util_malloc(num_pad_keys * sizeof * padded_keys,
                                    __func__);
  for(int key_nr = 0; key_nr < num_pad_keys; key_nr++)
  {
    assert(pad_keys[key_nr] != NULL);

    int key_len = strlen(pad_keys[key_nr]);
    padded_keys[key_nr] = util_malloc((key_len + 3) * sizeof * padded_keys[key_nr],
                                       __func__);
    padded_keys[key_nr][0] = ' ';
    for(int i=0; i<key_len; i++)
    {
      padded_keys[key_nr][1+i] = pad_keys[key_nr][i];
    }
    padded_keys[key_nr][key_len + 1] = ' ';
    padded_keys[key_nr][key_len + 2] = '\0';
  }
  char * buffer = util_string_replacen_alloc(buffer_wrk, num_pad_keys,
                                             pad_keys, (const char **) padded_keys);
  free(buffer_wrk);
  util_free_stringlist(padded_keys, num_pad_keys);

  return buffer;
}



char * conf_util_fscanf_alloc_token_buffer(
  const char * file_name)
{
  char * pad_keys[] = {"{","}","=",";","(",")"};
  char * buffer = __conf_util_fscanf_alloc_token_buffer(file_name, "--", 6, (const char **) pad_keys);
  return buffer;
}



/*
  This function takes a pointer to a position in a string and returns a copy
  of the next token in the string. Furthermore, the position in the string is
  moved to the position after the token. Strings inside quotations " and ' are
  treated as one token, and the quotations removed.
  
  Note: Quoted strings with no content, e.g. "   "  are NOT considered as tokens!

*/
char * conf_util_alloc_next_token(
  char ** buff_pos)
{
  char * sep = " \t\r\n,";

  int len_token = 0;
  bool found    = false;
  bool quoted   = false;
  while(!found)
  {
    int init_whitespace = strspn(*buff_pos, sep);
    *buff_pos += init_whitespace;

    if(*buff_pos[0] == '"')
    {
      quoted = true;
      *buff_pos += 1;
      len_token = strcspn(*buff_pos, "\"");
      if(len_token == strspn(*buff_pos, sep))
      {
        *buff_pos += len_token;
        len_token=0;
      }
    }
    else if(*buff_pos[0] == '\'')
    {
      quoted = true;
      *buff_pos += 1;
      len_token = strcspn(*buff_pos, "\'");
      if(len_token == strspn(*buff_pos, sep))
      {
        *buff_pos += len_token;
        len_token=0;
      }
    }
    else
    {
      quoted = false;
      len_token = strcspn(*buff_pos, sep);
    }

    if(len_token > 0)
      found = true;
    else if(len_token == 0 && !quoted)
      return NULL;
    else if(len_token == 0 && quoted)
      *buff_pos += 1;
  }

  char * token = util_malloc( (len_token + 1) * sizeof * token, __func__);
  memmove(token, *buff_pos, len_token);
  token[len_token] = '\0';
  *buff_pos += len_token;
  
  if(quoted)
    *buff_pos += 1;

  return token;
}
