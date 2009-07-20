#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <sched_util.h>
#include <buffer.h>

/**
   This file implements small utility functions used by the rest of
   the libsched library.
*/


static const int strip_comment = 1;
static const int strip_space   = 2;


static char * strip_line_alloc(const char * line) {
  const char  comment_char = '-';
  const char *space   = " \t";
  const int strip_mode = strip_comment + strip_space;
  char * new_line = NULL;
  int offset, length,pos;
  bool cont , quote_on , dash_on;   /* The quote_on variable protects againts premature termination based on '/' in path specifications. */
  
  if (strip_mode & strip_space)
    offset   = strspn(line , space);
  else
    offset = 0;

  dash_on  = false;
  quote_on = false;
  cont     = true;
  length   = 0;
  if (line[offset] != '\0') {
    pos = offset;
    do {
      if (line[pos] == '\'' || line[pos] == '"')
       quote_on = !quote_on;
      
      if (strip_mode & strip_comment) {
       if (!quote_on) {
         if (line[pos] == comment_char) {
           if (dash_on) {
             cont   = false;
             length = pos - offset - 1;
           } else 
             dash_on = true;
         } else
           dash_on = false;
       }
      }

      if (cont) {
       if (pos == (strlen(line) - 1)) {
         length = pos - offset + 1;
         cont = false;
       }
      }
      
      if (cont)
       pos++;
    } while (cont);

    /*
      Remove trailing space:
    */

    if (strip_mode & strip_space) {
      if (offset + length > 0) {
       while (line[offset + length - 1] == ' ') 
         length--;
      }
    }

    if (length > 0) 
      new_line = util_realloc_substring_copy(NULL , &line[offset] , length);
    else 
      new_line = NULL;
    
  } 
  
  return new_line;
}


/*****************************************************************/



char * sched_util_alloc_line(FILE *stream , bool *at_eof) {
  char *tmp_line = util_fscanf_alloc_line(stream , at_eof);
  char * line    = strip_line_alloc( tmp_line );
  
  free(tmp_line);
  return line;
}



void sched_util_parse_file(const char *filename , int *_lines , char ***_line_list) {
  int    lines;
  char **line_list;
  int    buffer_lines;
  bool   at_eof;
  FILE *stream = util_fopen(filename , "r");
  
  lines        = 0;
  buffer_lines = 100;
  line_list    = NULL;
  
  line_list = realloc(line_list , buffer_lines * sizeof *line_list);
  do {
    char * line = sched_util_alloc_line(stream , &at_eof);
    if (line != NULL) {
      line_list[lines] = line;
      lines++;

      if (lines == buffer_lines) {
       buffer_lines *= 2;
       line_list = realloc(line_list , buffer_lines * sizeof *line_list);
      }
    }
  } while (!at_eof);
  fclose(stream);

  line_list = realloc(line_list , lines * sizeof *line_list);
  *_lines     = lines;
  *_line_list = line_list;
}






/**
   Added delimiter on "'" - not well tested.
*/

void sched_util_parse_line(const char * line , int *_tokens , char ***_token_list , int min_tokens , bool *slash_term) {
  const char *delimiters = " \'";
  int    token,tokens,offset,length;
  char **token_list;
  
  tokens  = 0;
  offset = strspn(line , delimiters);

  while (line[offset ] != '\0' && line[offset] != '/') {
    length = strcspn(&line[offset] , delimiters);
    if (length > 0) {
      char * token_string = util_alloc_substring_copy(&line[offset] , length);
      char * star_ptr     = token_string;
      long int items      = strtol(token_string , &star_ptr , 10);
      
      if (star_ptr != token_string && star_ptr[0] == '*') 
       tokens += items;
      else
       tokens++;
      
      offset += length;                              /* Skipping the token */
      offset += strspn(&line[offset] , delimiters);  /* Skipping the following whitespace */

      free(token_string);
    }
  }
  
  if (tokens < min_tokens)
    tokens = min_tokens;
  token_list = util_malloc(tokens * sizeof *token_list , __func__);
  for (token = 0; token < tokens; token++) 
    token_list[token] = NULL;
  
  token   = 0;
  offset = strspn(line , delimiters);
  while (line[offset ] != '\0' && line[offset] != '/' ) {
    length = strcspn(&line[offset] , delimiters);
    if (length > 0) {
      char * token_string  = util_alloc_substring_copy(&line[offset] , length);
      char * star_ptr      = token_string;
      long int items = strtol(token_string , &star_ptr , 10);
      
      if (star_ptr != token_string && star_ptr[0] == '*') {
       int t1 = token;
       int t2 = token + items;
           
       for (token = t1; token < t2; token++) {
         if (strlen(star_ptr) == 1)
           token_list[token] = NULL;
         else
           token_list[token] = strdup(&star_ptr[1]);
       }
      } else {
       token_list[token] = util_alloc_substring_copy(&line[offset] , length);
       token++;
      }
      
      offset += length;                              /* Skipping the token */
      offset += strspn(&line[offset] , delimiters);  /* Skipping the following whitespace */
      free(token_string);
    }
  }
  if (slash_term != NULL) {
    if (line[offset] == '/')
      *slash_term = true;
    else
      *slash_term = false;
  }
  
  /*
    Removing quotes ...
  */
  for (token = 0; token < tokens; token++) {
    if (token_list[token] != NULL) {
      if (token_list[token][0] == '\'') {
       char *tmp = util_alloc_substring_copy(&token_list[token][1] , strlen(token_list[token]) - 2);
       free(token_list[token]);
       token_list[token] = tmp;
      }
    }
  }

  *_tokens = tokens;
  *_token_list = token_list;
}


/*****************************************************************/
/**
   This function reads up to the next '/'. 

   It has been a quite strong assumption in this code that each line
   (in text-file meaning of the word) should be one complete '/'
   terminated record; however suddenly the following keyword comes in
   from the blue:

     RPTSCHED
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
         0 0 0 0 0 /

   I.e. the keyword has only one 'logical' line, however it spans
   several newlines in the file. This function reads file content up
   until the next '/'; the file content is read verbatim, no comment
   discarding or anything is performed.
 
    o If no  '/'  is found the function will fail HARD.
    o If the '/' is the first character the at_eokw variable is set to true.

*/
   

char * sched_util_alloc_slash_terminated_line(FILE * stream) {
  buffer_type * buffer = buffer_alloc( 256 );
  int c;
  buffer_clear( buffer );
  while (true) {
    c = fgetc(stream);
    if (c == EOF)
      util_abort("%s: fatal error when parsing SCHEDULE file. Line:%d - not terminated by: \'/\'\n",__func__ , util_get_current_linenr( stream ));
    buffer_fwrite_char( buffer , ( char ) c );
    if (c == '/')
      break;

  }
  buffer_fwrite_char( buffer , '\0' );  /* Ensure \0 termination. */
  {
    char * line = buffer_alloc_data_copy( buffer );
    buffer_free( buffer );
    return line;
  }
}


/*****************************************************************/

char * sched_util_alloc_next_entry(FILE * stream, bool * at_eof, bool * at_eokw)
{
  char * line;

  for(;;)
  {
    line = sched_util_alloc_line(stream, at_eof);
    if(line == NULL)
    {
      if(*at_eof)
      {
        return NULL;
      }
      else
      {
        continue;
      }
    }
    else
    {
      if(line[0] == '/')
      {
        *at_eokw = true;
        free(line);
        return NULL;
      }
      else
      {
        return line;
      }
    }
  }
}




void sched_util_fprintf_default(int width , FILE * stream) {
  fprintf(stream , "1*");
  for (int i=0; i < (width - 2); i++) 
    fputc(' ' , stream);
}

/**
   All the sched_util_fprintf_xxx() functions start by putting out one
   ' ', to ensure against overlapping fields.
*/


void sched_util_fprintf_dbl(bool def, double value , int width , int dec , FILE *stream) {
  fputc(' ' , stream);
  if (def) 
    sched_util_fprintf_default( width , stream);
  else 
    util_fprintf_double(value , width , dec , 'f' , stream);
}



void sched_util_fprintf_int(bool def, int value , int width , FILE *stream) {
  fputc(' ' , stream);
  if (def) {
    sched_util_fprintf_default( width , stream);
  } else 
    util_fprintf_int(value , width , stream);
}


/*
  The formatting is ridicolusly inflexible - don't touch this shit.
*/

void sched_util_fprintf_qst(bool def, const char *s , int width , FILE *stream) {
  fputc(' ' , stream);
  if (def) {
    sched_util_fprintf_default( width , stream );
  } else {
    for (int i=0; i < (width - strlen(s)); i++) 
      fputc(' ' , stream);
    
    fputc('\'' , stream);
    fprintf(stream , "%s" , s);
    fputc('\'' , stream);
  }
}




double sched_util_atof(const char *token) {
  if (token != NULL) {
    double value;
    if (!util_sscanf_double(token , &value))
      util_abort("%s: failed to parse:\"%s\" as floating point number. \n",__func__ , token);
    return value;
  } else
    return 0.0;
}


int sched_util_atoi(const char *token) {
  if (token != NULL) {
    int value;
    if (!util_sscanf_int(token , &value))
      util_abort("%s: failed to parse:\"%s\" as integer \n",__func__ , token);
    return value;
  } else
    return 0;
}



/* Simple utility function used for debugging. */
void sched_util_fprintf_tokenlist(int num_token , const char ** token_list , const bool * def) {
  int i;
  for (i = 0; i < num_token; i++) {
    if (def[i])
      fprintf(stdout , " \'*\' " );
    else
      fprintf(stdout , " \'%s\' " , token_list[i]);
  }
  fprintf(stdout , "\n");
}




