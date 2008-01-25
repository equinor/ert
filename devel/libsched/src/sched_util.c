#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <sched_util.h>

static const int strip_comment = 1;
static const int strip_space   = 2;


static char * strip_line_alloc(const char * line) {
  const char  comment_char = '-';
  const char *space   = " \t";
  const int strip_mode = strip_comment + strip_space;
  char * new_line = NULL;
  int offset, length,pos;
  bool cont , quote_on , dash_on;
  
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



static char * alloc_line(FILE *stream , bool *at_eof) {
  char *tmp_line = util_fscanf_alloc_line(stream , at_eof);
  char * line    = strip_line_alloc( tmp_line );
  
  free(tmp_line);
  return line;
}




/*****************************************************************/

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
    char * line = alloc_line(stream , &at_eof);
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




void sched_util_free_token_list(int size, char **token_list) {
  int i;
  for (i=0; i < size; i++) 
    if (token_list[i] != NULL) free(token_list[i]);


  free(token_list);
}


void sched_util_parse_line(const char * line , int *_tokens , char ***_token_list , int min_tokens , bool *slash_term) {
  const char *delimiters = " ";
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
  token_list = malloc(tokens * sizeof *token_list);
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


void sched_util_fprintf_dbl(bool def, double value , int width , int dec , FILE *stream) {
  if (def) {
    int i;
    fprintf(stream , " 1*");
    for (i=0; i < (width - 2); i++) 
      fputc(' ' , stream);
  } else {
    char fmt[16];
    fmt[0] = '%';
    sprintf(&fmt[1] , "%d.%df " , width , dec);
    fprintf(stream , fmt , value);
  }
}



void sched_util_fprintf_int(bool def, int value , int width , FILE *stream) {
  if (def) {
    int i;
    fprintf(stream , " 1*");
    for (i=0; i < (width - 2); i++) 
      fputc(' ' , stream);
  } else {
    char fmt[16];
    fmt[0] = '%';
    sprintf(&fmt[1] , "%dd " , width);
    fprintf(stream , fmt , value);
  }
}


/*
  The formatting is ridicolusly inflexible - don't touch this shit.
*/

void sched_util_fprintf_qst(bool def, const char *s , int width , FILE *stream) {
  int i;
  if (def) {
    fprintf(stream , "  1*        ");
    /*
      for (i=0; i < (width); i++) 
      fputc(' ' , stream);
    */
  } else {

    for (i=0; i < (width + 1 - strlen(s)); i++) 
      fputc(' ' , stream);
    
    fputc('\'' , stream);
    fprintf(stream , "%s" , s);
    fputc('\'' , stream);
  }
}




double sched_util_atof(const char *token) {
  if (token != NULL)
    return atof(token);
  else
    return 0.0;
}


int sched_util_atoi(const char *token) {
  if (token != NULL)
    return atoi(token);
  else
    return 0;
}

      

void sched_util_fprintf_days_line(int date_nr , time_t t1 , time_t t2 , FILE *stream) {
  struct tm ts;
  double days;
  localtime_r(&t2 , &ts);
  days = difftime(t2 , t1) / (24 * 3600);
  fprintf(stream , "%04d  %8.2f %4d %2d %2d\n",date_nr , days , ts.tm_year + 1900 , ts.tm_mon + 1, ts.tm_mday);
}

