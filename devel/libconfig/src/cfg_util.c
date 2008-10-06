#include <util.h>
#include <string.h>

char * cfg_util_fscanf_alloc_token(FILE * stream, bool * sub_start, bool * sub_end, bool * at_eof)
{
  char * token = util_fscanf_alloc_token(stream);
  if(token == NULL)
  {
    util_forward_line(stream, at_eof);
    return NULL;
  }
  else
  {
    int token_len = strlen(token);

    if(strcmp(token, "--") == 0)
    {
      free(token);
      util_forward_line(stream, at_eof); 
      return NULL;
    }
    if(token_len > 1)
    {
      if(util_char_in('{', strlen(token), token))
      {
        util_abort("%s: Found bracket { in token %s. Brackets shall be strictly separated from tokens.\n", __func__, token);
        return NULL;
      }
      else if(util_char_in('}', strlen(token), token))
      {
        util_abort("%s: Found bracket } in token %s. Brackets shall be strictly separated from tokens.\n", __func__, token);
        return NULL;
      }
      else
      {
        return token;
      }
    }
    else
    {
      if(token[0] == '{')
      {
        *sub_start = true;
        free(token);
        return NULL;
      }
      else if(token[0] == '}')
      {
        *sub_end = true;
        free(token);
        return NULL;
      }
      else
      {
        return token;
      }
    }
  }
}
