#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <enkf_site_config.h>


int main(void) {
  enkf_site_config_type * site = enkf_site_config_bootstrap("site-config");
  
  enkf_site_config_free(site);
}



