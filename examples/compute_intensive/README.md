## Compute intensive example
The purpose of this example is to test both cost of compute time and performance of disk operations.

### How to run
See run_test

``` sh
ert3 init

ert3 record load hash_input resources/hash_input.json
ert3 run compute_test

ert3 record load io_test_input resources/io_test_input.json
ert3 run io_test
```


### Compute test
The compute test creates hashes until a specified timeout. The function is defined in hash.py and the input data is defined in resources/hash_input.json. Below is an example of the input record:  

``` json
{
    "timeout_seconds" : 100
}
```



### IO test
The io test writes random bytes to a file on the local filesystem. The function is defined in read_write.py and the input data is defined in resources/io_test_input.json. Below is an example of the input record:

``` json
{
    "num_files" : 3,
    "file_size" : 1073741824  
}
```

The files size is specified in bytes.
