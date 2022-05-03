/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'block_fs.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <errno.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fmt/ostream.h>

#include <ert/python.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/vector.hpp>

#include <ert/res_util/block_fs.hpp>

namespace fs = std::filesystem;

#define MOUNT_MAP_MAGIC_INT 8861290
#define BLOCK_FS_TYPE_ID 7100652

/*
  During mounting a significant part of the time is spent on filling
  up the index hash table. By default a hash table is created with a
  quite small size, and when initializing a large block_fs structure
  it must be resized many times. By setting a default size with the
  DEFAULT_INDEX_SIZE variable the hash table will immediately be
  resized, avoiding some of the automatic calls to hash_resize.

  When the file system is loaded from an index a good size estimate
  can be inferred directly from the index.
*/

#define DEFAULT_INDEX_SIZE 2048

/*
   These should be bitwise "smart" - so it is possible
   to go on a wild chase through a binary stream and look for them.
*/

/** Binary(85)  =  01010101 */
#define NODE_IN_USE_BYTE 85
#define WRITE_START__ 77162

static const int NODE_END_TAG =
    16711935; /* Binary      =  00000000111111110000000011111111 */
static const int NODE_WRITE_ACTIVE_START = WRITE_START__;
static const int NODE_WRITE_ACTIVE_END = 776512;

typedef enum {
    /** NODE_IN_USE_BYTE * ( 1 + 256 + 256**2 + 256**3) => Binary 01010101010101010101010101010101 */
    NODE_IN_USE = 1431655765,
    NODE_WRITE_ACTIVE = WRITE_START__, /* This */
    /** This should __never__ be written to disk */
    NODE_INVALID = 13
} node_status_type;

typedef struct file_node_struct file_node_type;

struct file_node_struct {
    /** The offset into the data_file of this node. NEVER Changed. */
    long int node_offset;
    /** The offset from the node start to the start of actual data - i.e. data
     * starts at absolute position: node_offset + data_offset. */
    int data_offset;
    /** The size in bytes of this node - must be >= data_size. NEVER Changed. */
    int node_size;
    /** The size of the data stored in this node - in addition the node might
     * need to store header information. */
    int data_size;
    /** This should be: NODE_IN_USE; in addition the disk can have
     * NODE_WRITE_ACTIVE for incomplete writes. */
    node_status_type status;
};

struct block_fs_struct {
    UTIL_TYPE_ID_DECLARATION;

    int data_fd;
    FILE *data_stream;

    /** The total number of bytes in the data_file. */
    long int data_file_size;

    std::mutex mutex;

    /** THE HASH table of all the nodes/files which have been stored. */
    hash_type *index;
    /** This vector owns all the file_node instances - the index structure only
     * contain pointers to the objects stored in this vector. */
    vector_type *file_nodes;
    /** This just counts the number of writes since the file system was mounted. */
    bool data_owner;
};

UTIL_SAFE_CAST_FUNCTION(block_fs, BLOCK_FS_TYPE_ID)

static inline void fseek__(FILE *stream, long int arg, int whence) {
    if (fseek(stream, arg, whence) != 0) {
        fprintf(stderr, "** Warning - seek:%ld failed %s(%d) \n", arg,
                strerror(errno), errno);
        util_abort("%S - aborting\n", __func__);
    }
}

static inline void block_fs_fseek(block_fs_type *block_fs, long offset) {
    fseek__(block_fs->data_stream, offset, SEEK_SET);
}

/**
   Observe that the two input arguments to this function should NEVER
   change. They represent offset and size in the underlying data file,
   and that is for ever fixed.
*/
static file_node_type *file_node_alloc(node_status_type status, long int offset,
                                       int node_size) {
    file_node_type *file_node =
        (file_node_type *)util_malloc(sizeof *file_node);

    file_node->node_offset = offset;  /* These should NEVER change. */
    file_node->node_size = node_size; /* -------------------------  */

    file_node->data_size = 0;
    file_node->data_offset = 0;
    file_node->status = status;

    return file_node;
}

static void file_node_free(file_node_type *file_node) { free(file_node); }

static void file_node_free__(void *file_node) {
    file_node_free((file_node_type *)file_node);
}

static bool file_node_verify_end_tag(const file_node_type *file_node,
                                     FILE *stream) {
    int end_tag;
    fseek__(stream,
            file_node->node_offset + file_node->node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    if (fread(&end_tag, sizeof end_tag, 1, stream) == 1) {
        if (end_tag == NODE_END_TAG)
            return true;
        else
            return false;
    } else
        return false;
}

static file_node_type *file_node_fread_alloc(FILE *stream, char **key) {
    file_node_type *file_node = NULL;
    node_status_type status;
    long int node_offset = ftell(stream);
    if (fread(&status, sizeof status, 1, stream) == 1) {
        if (status == NODE_IN_USE) {
            int node_size;
            *key = util_fread_realloc_string(*key, stream);

            node_size = util_fread_int(stream);
            if (node_size <= 0)
                status = NODE_INVALID;
            // A case has occured with an invalid node with size 0. That
            // resulted in a deadlock, because the reader never got beyond
            // the broken node. We therefore explicitly check for this
            // condition.
            file_node = file_node_alloc(status, node_offset, node_size);
            if (status == NODE_IN_USE) {
                file_node->data_size = util_fread_int(stream);
                file_node->data_offset = ftell(stream) - file_node->node_offset;
            }
        } else {
            // We did not recognize the status identifier; the node will
            // eventually be marked as free.
            if (status != NODE_WRITE_ACTIVE)
                status = NODE_INVALID;
            file_node = file_node_alloc(status, node_offset, 0);
        }
    }
    return file_node;
}

/*
   Internal index layout:

   |<InUse: Bool><Key: String><node_size: Int><data_size: Int>|
   |<InUse: Bool><node_size: Int><data_size: Int>|

  /|\
   |
   |<-------------------------------------------------------->|
                                                   |
node_offset                                      offset

  The node_offset and offset values are not stored on disk, but rather
  implicitly read with ftell() calls.
*/

/**
   This function will write the node information to file, this
   includes the NODE_END_TAG identifier which shoule be written to the
   end of the node.
*/
static void file_node_fwrite(const file_node_type *file_node, const char *key,
                             FILE *stream) {
    if (file_node->node_size == 0)
        util_abort("%s: trying to write node with z<ero size \n", __func__);
    {
        fseek__(stream, file_node->node_offset, SEEK_SET);
        util_fwrite_int(file_node->status, stream);
        if (file_node->status == NODE_IN_USE)
            util_fwrite_string(key, stream);
        util_fwrite_int(file_node->node_size, stream);
        util_fwrite_int(file_node->data_size, stream);
        fseek__(stream,
                file_node->node_offset + file_node->node_size -
                    sizeof NODE_END_TAG,
                SEEK_SET);
        util_fwrite_int(NODE_END_TAG, stream);
    }
}

/**
   This marks the start and end of the node with the integer tags:
   NODE_WRITE_ACTIVE_START and NODE_WRITE_ACTIVE_END, signalling this
   section in the data file is 'work in progress', and should be
   discarded if the application aborts during the write.

   When the write is complete file_node_fwrite() should be called,
   which will replace the NODE_WRITE_ACTIVE_START and
   NODE_WRITE_ACTIVE_END tags with NODE_IN_USE and NODE_END_TAG
   identifiers.
*/
static void file_node_init_fwrite(const file_node_type *file_node,
                                  FILE *stream) {
    fseek__(stream, file_node->node_offset, SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_START, stream);
    fseek__(stream,
            file_node->node_offset + file_node->node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_END, stream);
}

/**
   Observe that header in this context includes the size of the tail
   marker NODE_END_TAG.
*/
static int file_node_header_size(const char *filename) {
    file_node_type *file_node;
    return sizeof(file_node->status) + sizeof(file_node->node_size) +
           sizeof(file_node->data_size) + sizeof(NODE_END_TAG) +
           sizeof(int) /* embedded by the util_fwrite_string routine */ +
           strlen(filename) + 1 /* \0 */;
}

static void file_node_set_data_offset(file_node_type *file_node,
                                      const char *filename) {
    file_node->data_offset =
        file_node_header_size(filename) - sizeof(NODE_END_TAG);
}

static void block_fs_insert_index_node(block_fs_type *block_fs,
                                       const char *filename,
                                       const file_node_type *file_node) {
    hash_insert_ref(block_fs->index, filename, file_node);
}

/**
   Installing the new node AND updating file tail.
*/
static void block_fs_install_node(block_fs_type *block_fs,
                                  file_node_type *node) {
    block_fs->data_file_size =
        (block_fs->data_file_size > (node->node_offset + node->node_size))
            ? block_fs->data_file_size
            : node->node_offset + node->node_size;
    vector_append_owned_ref(block_fs->file_nodes, node, file_node_free__);
}

static void block_fs_reinit(block_fs_type *block_fs) {
    block_fs->index = hash_alloc();
    block_fs->file_nodes = vector_alloc_new();
    block_fs->data_file_size = 0;
}

static block_fs_type *block_fs_alloc_empty(const fs::path &mount_file,
                                           bool read_only) {
    block_fs_type *block_fs = new block_fs_type;
    UTIL_TYPE_ID_INIT(block_fs, BLOCK_FS_TYPE_ID);

    FILE *stream = util_fopen(mount_file.c_str(), "r");
    int id = util_fread_int(stream);
    int version = util_fread_int(stream);
    if (version != 0)
        throw std::runtime_error(fmt::format(
            "block_fs data version unexpected. Expected 0, got {}", version));

    fclose(stream);

    if (id != MOUNT_MAP_MAGIC_INT)
        util_abort("%s: The file:%s does not seem to be a valid block_fs "
                   "mount map \n",
                   __func__, mount_file.c_str());

    block_fs_reinit(block_fs);

    block_fs->data_owner = !read_only;
    return block_fs;
}

UTIL_IS_INSTANCE_FUNCTION(block_fs, BLOCK_FS_TYPE_ID);

static void block_fs_fwrite_mount_info(const fs::path &mount_file) {
    FILE *stream = util_fopen(mount_file.c_str(), "w");
    util_fwrite_int(MOUNT_MAP_MAGIC_INT, stream);
    util_fwrite_int(0 /* data version; unused */, stream);
    fclose(stream);
}

/**
   Will seek the datafile to the end of the current file_node. So that the next read will be "guaranteed" to
   start at a new node.
*/
static void block_fs_fseek_node_end(block_fs_type *block_fs,
                                    const file_node_type *file_node) {
    block_fs_fseek(block_fs, file_node->node_offset + file_node->node_size);
}

static void block_fs_fseek_node_data(block_fs_type *block_fs,
                                     const file_node_type *file_node) {
    block_fs_fseek(block_fs, file_node->node_offset + file_node->data_offset);
}

/**
   This function will read through the datafile seeking for the identifier:
   NODE_IN_USE. If the valid status identifier is found the stream is
   repositioned at the beginning of the valid node.

   If no valid status ID is found whatsover the data_stream
   indicator is left at the end of the file; and the calling scope
   will finish from there.
*/
static bool block_fs_fseek_valid_node(block_fs_type *block_fs) {
    unsigned char byte;
    int status;
    while (true) {
        if (fread(&byte, sizeof byte, 1, block_fs->data_stream) == 1) {
            if (byte == NODE_IN_USE_BYTE) {
                long int pos = ftell(block_fs->data_stream);
                // OK - we found one interesting byte; let us try to read the
                // whole integer and see if we have hit any of the valid status
                // identifiers.
                fseek__(block_fs->data_stream, -1, SEEK_CUR);
                if (fread(&status, sizeof status, 1, block_fs->data_stream) ==
                    1) {
                    if (status == NODE_IN_USE) {
                        // OK - we have found a valid identifier. We reposition
                        // to the start of this valid status id and return
                        // true.
                        fseek__(block_fs->data_stream, -sizeof status,
                                SEEK_CUR);
                        return true;
                    } else
                        // OK - this was not a valid id; we go back and
                        // continue reading single bytes.
                        block_fs_fseek(block_fs, pos);
                } else
                    break; /* EOF */
            }
        } else
            break; /* EOF */
    }
    fseek__(block_fs->data_stream, 0, SEEK_END);
    return false;
}

/**
   The read-only open mode is only for the mount section, where the
   data file is read in to load/verify the index.

   If the read_only open fails - the data_stream is set to NULL. If
   the open succeeds the calling scope should close the stream before
   calling this function again, with read_only == false.
*/
static void block_fs_open_data(block_fs_type *block_fs,
                               const fs::path &data_file) {
    if (!block_fs_is_readonly(block_fs)) {
        /* Normal read-write open.- */
        if (fs::exists(data_file))
            block_fs->data_stream = util_fopen(data_file.c_str(), "r+");
        else
            block_fs->data_stream = util_fopen(data_file.c_str(), "w+");
    } else {
        /* read-only open. */
        if (fs::exists(data_file.c_str()))
            block_fs->data_stream = util_fopen(data_file.c_str(), "r");
        else
            block_fs->data_stream = NULL;
        // If we ever try to dereference this pointer it will break
        // hard; but it should be stopped in hash_get() calls before the
        // data_stream is dereferenced anyway?
    }
    if (block_fs->data_stream == NULL)
        block_fs->data_fd = -1;
    else
        block_fs->data_fd = fileno(block_fs->data_stream);
}

/**
   This function will 'fix' the nodes with offset in offset_list.  The
   fixing in this case means the following:

     1. The node is updated in place on the file to become a free node.
     2. The node is added to the block_fs instance as a free node, which can
        be recycled at a later stage.

   If the instance is read-only the function
   will return immediately.
*/
static void block_fs_fix_nodes(block_fs_type *block_fs,
                               const std::vector<long> &offset_list) {
    if (!block_fs_is_readonly(block_fs)) {
        fsync(block_fs->data_fd);

        char *key = NULL;
        for (const auto &node_offset : offset_list) {
            bool new_node = false;
            file_node_type *file_node;
            block_fs_fseek(block_fs, node_offset);
            file_node = file_node_fread_alloc(block_fs->data_stream, &key);

            if ((file_node->status == NODE_INVALID) ||
                (file_node->status == NODE_WRITE_ACTIVE)) {
                /* This node is really quite broken. */
                long int node_end;
                block_fs_fseek_valid_node(block_fs);
                node_end = ftell(block_fs->data_stream);
                file_node->node_size = node_end - node_offset;
            }

            file_node->status = NODE_INVALID;
            file_node->data_size = 0;
            file_node->data_offset = 0;

            block_fs_fseek(block_fs, node_offset);
            file_node_fwrite(file_node, NULL, block_fs->data_stream);
            if (!new_node)
                file_node_free(file_node);
        }
        free(key);

        fsync(block_fs->data_fd);
    }
}

static void block_fs_build_index(block_fs_type *block_fs,
                                 const fs::path &data_file,
                                 std::vector<long> &error_offset) {
    char *filename = NULL;
    file_node_type *file_node;

    hash_resize(block_fs->index, DEFAULT_INDEX_SIZE);
    block_fs_fseek(block_fs, 0);
    do {
        file_node = file_node_fread_alloc(block_fs->data_stream, &filename);
        if (file_node != NULL) {
            if ((file_node->status == NODE_INVALID) ||
                (file_node->status == NODE_WRITE_ACTIVE)) {
                if (file_node->status == NODE_INVALID)
                    fprintf(stderr,
                            "** Warning:: invalid node found at offset:%ld in "
                            "datafile:%s - data will be lost, node_size:%d\n",
                            file_node->node_offset, data_file.c_str(),
                            file_node->node_size);
                else
                    fprintf(
                        stderr,
                        "** Warning:: file system was prematurely shut down "
                        "while writing node in %s/%ld - will be discarded.\n",
                        data_file.c_str(), file_node->node_offset);

                error_offset.push_back(file_node->node_offset);
                file_node_free(file_node);
                block_fs_fseek_valid_node(block_fs);
            } else {
                if (file_node_verify_end_tag(file_node,
                                             block_fs->data_stream)) {
                    block_fs_fseek_node_end(block_fs, file_node);
                    block_fs_install_node(block_fs, file_node);
                    if (file_node->status == NODE_IN_USE) {
                        block_fs_insert_index_node(block_fs, filename,
                                                   file_node);
                    } else {
                        util_abort("%s: node status flag:%d not recognized - "
                                   "error in data file \n",
                                   __func__, file_node->status);
                    }
                } else {
                    // Could not find a valid END_TAG - indicating that the
                    // filesystem was shut down during the write of this node.
                    // This node will NOT be added to the index.  The node will
                    // be updated to become a free node.
                    fprintf(stderr,
                            "** Warning found node:%s at offset:%ld which was "
                            "incomplete - discarded.\n",
                            filename, file_node->node_offset);
                    error_offset.push_back(file_node->node_offset);
                    file_node_free(file_node);
                    block_fs_fseek_valid_node(block_fs);
                }
            }
        }
    } while (file_node != NULL);
    free(filename);
}

static bool block_fs_is_readonly(const block_fs_type *bfs) {
    if (bfs->data_owner)
        return false;
    else
        return true;
}

block_fs_type *block_fs_mount(const fs::path &mount_file, bool read_only) {
    fs::path path = mount_file.parent_path();
    std::string base_name = mount_file.stem();
    auto data_file = path / (base_name + ".data_0");
    auto index_file = path / (base_name + ".index");
    block_fs_type *block_fs;
    if (!fs::exists(mount_file))
        /* This is a brand new filesystem - create the mount map first. */
        block_fs_fwrite_mount_info(mount_file);
    std::vector<long> fix_nodes;
    block_fs = block_fs_alloc_empty(mount_file, read_only);

    block_fs_open_data(block_fs, data_file);
    if (block_fs->data_stream != nullptr) {
        std::error_code ec;
        fs::remove(index_file, ec /* error code is ignored */);
        block_fs_build_index(block_fs, data_file, fix_nodes);
    }
    block_fs_fix_nodes(block_fs, fix_nodes);
    return block_fs;
}

static file_node_type *block_fs_get_new_node(block_fs_type *block_fs,
                                             const char *filename,
                                             size_t size) {

    long int offset;
    file_node_type *new_node;

    /* Must lock the total size here ... */
    offset = block_fs->data_file_size;
    new_node = file_node_alloc(NODE_IN_USE, offset, size);
    block_fs_install_node(
        block_fs, new_node); /* <- This will update the total file size. */

    return new_node;
}

bool block_fs_has_file(block_fs_type *block_fs, const char *filename) {
    std::lock_guard guard{block_fs->mutex};
    return hash_has_key(block_fs->index, filename);
}

/**
   It seems it is not enough to call fsync(); must also issue this
   funny fseek + ftell combination to ensure that all data is on
   disk after an uncontrolled shutdown.

   Could possibly use fdatasync() to improve speed slightly?
*/
void block_fs_fsync(block_fs_type *block_fs) {
    if (!block_fs_is_readonly(block_fs)) {
        fsync(block_fs->data_fd);
        block_fs_fseek(block_fs, block_fs->data_file_size);
        ftell(block_fs->data_stream);
    }
}

/**
   The single lowest-level write function:

   3. seek to correct position.
   4. Write the data with util_fwrite()

   8. set the data_size field of the node.

   Observe that when 'designing' this file-system the priority has
   been on read-spead, one consequence of this is that all write
   operations are sandwiched between two fsync() calls; that
   guarantees that the read access (which should be the fast path) can
   be without any calls to fsync().

   Not necessary to lock - since all writes are protected by the
   'global' rwlock anyway.
*/
static void block_fs_fwrite__(block_fs_type *block_fs, const char *filename,
                              file_node_type *node, const void *ptr,
                              int data_size) {
    block_fs_fseek(block_fs, node->node_offset);
    node->status = NODE_IN_USE;
    node->data_size = data_size;
    file_node_set_data_offset(node, filename);

    // This marks the node section in the datafile as write in progress with:
    // NODE_WRITE_ACTIVE_START ... NODE_WRITE_ACTIVE_END
    file_node_init_fwrite(node, block_fs->data_stream);

    /* Writes the actual data content. */
    block_fs_fseek_node_data(block_fs, node);
    util_fwrite(ptr, 1, data_size, block_fs->data_stream, __func__);

    /* Writes the file node header data, including the NODE_END_TAG. */
    file_node_fwrite(node, filename, block_fs->data_stream);
}

void block_fs_fwrite_file(block_fs_type *block_fs, const char *filename,
                          const void *ptr, size_t data_size) {
    if (block_fs_is_readonly(block_fs))
        throw std::runtime_error("tried to write to read only filesystem");
    std::lock_guard guard{block_fs->mutex};

    file_node_type *file_node;
    bool new_node = true;
    size_t min_size = data_size + file_node_header_size(filename);

    file_node = block_fs_get_new_node(block_fs, min_size);

    /* The actual writing ... */
    block_fs_fwrite__(block_fs, filename, file_node, ptr, data_size);
    if (new_node)
        block_fs_insert_index_node(block_fs, filename, file_node);
}

void block_fs_fwrite_buffer(block_fs_type *block_fs, const char *filename,
                            const buffer_type *buffer) {
    block_fs_fwrite_file(block_fs, filename, buffer_get_data(buffer),
                         buffer_get_size(buffer));
}

/**
   Reads the full content of 'filename' into the buffer.
*/
void block_fs_fread_realloc_buffer(block_fs_type *block_fs,
                                   const char *filename, buffer_type *buffer) {
    std::lock_guard guard{block_fs->mutex};
    file_node_type *node =
        (file_node_type *)hash_get(block_fs->index, filename);

    buffer_clear(buffer); /* Setting: content_size = 0; pos = 0;  */

    block_fs_fseek_node_data(block_fs, node);
    buffer_stream_fread(buffer, node->data_size, block_fs->data_stream);

    buffer_rewind(buffer); /* Setting: pos = 0; */
}

/**
   Close/synchronize the open file descriptors and free all memory
   related to the block_fs instance.

*/
void block_fs_close(block_fs_type *block_fs) {
    block_fs_fsync(block_fs);

    if (block_fs->data_stream != NULL)
        fclose(block_fs->data_stream);

    hash_free(block_fs->index);
    vector_free(block_fs->file_nodes);
    delete block_fs;
}
