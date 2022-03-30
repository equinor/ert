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
#include <optional>
#include <stdexcept>
#include <vector>
#include <mutex>

#include <fmt/ostream.h>

#include <unordered_map>
#include <ert/util/buffer.hpp>

#include <ert/res_util/block_fs.hpp>

namespace fs = std::filesystem;

#define MOUNT_MAP_MAGIC_INT 8861290
#define BLOCK_FS_TYPE_ID 7100652

/*
   These should be bitwise "smart" - so it is possible
   to go on a wild chase through a binary stream and look for them.
*/

#define NODE_IN_USE_BYTE 85 /* Binary(85)  =  01010101 */
#define WRITE_START__ 77162

static const int NODE_END_TAG =
    16711935; /* Binary      =  00000000111111110000000011111111 */
static const int NODE_WRITE_ACTIVE_START = WRITE_START__;
static const int NODE_WRITE_ACTIVE_END = 776512;

typedef enum {
    NODE_IN_USE = 0x55555555,
    NODE_WRITE_ACTIVE = WRITE_START__, /* This */
    NODE_INVALID = 13 /* This should __never__ be written to disk */
} node_status_type;

typedef struct file_node_struct file_node_type;

/*
  Datastructure representing one 'block' in the datafile. The block
  can either refer to a file (status == NODE_IN_USE) or to an empty
  slot in the datafile (status == NODE_FREE).
*/

struct file_node_struct {
    long int node_offset{};
    int node_size{};
    int data_offset{};
    int data_size{};
    node_status_type status;

    file_node_struct() = default;
    file_node_struct(const file_node_struct &) = default;
    file_node_struct(node_status_type status, long int node_offset,
                     int node_size)
        : node_offset(node_offset), node_size(node_size), status(status) {}
};

struct block_fs_struct {
    UTIL_TYPE_ID_DECLARATION;
    fs::path mount_path;
    std::string base_name;

    fs::path data_path() const { return mount_path / (base_name + ".data_0"); }

    fs::path lock_path() const { return mount_path / (base_name + ".lock_0"); }

    FILE *data_stream;

    int lock_fd; /* The file descriptor for the lock_file. Set to -1 if we do not have write access. */

    mutable std::recursive_mutex mutex;

    std::unordered_map<std::string, file_node_type> index;
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

static bool file_node_verify_end_tag(const file_node_type &file_node,
                                     FILE *stream) {
    int end_tag;
    fseek__(stream,
            file_node.node_offset + file_node.node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    if (fread(&end_tag, sizeof end_tag, 1, stream) == 1) {
        if (end_tag == NODE_END_TAG)
            return true; /* All hunkadory. */
        else
            return false;
    } else
        return false;
}

static std::optional<file_node_type> file_node_fread_alloc(FILE *stream,
                                                           char **key) {
    node_status_type status;
    long int node_offset = ftell(stream);
    if (fread(&status, sizeof status, 1, stream) == 1) {
        if (status == NODE_IN_USE) {
            int node_size;
            if (status == NODE_IN_USE)
                *key = util_fread_realloc_string(*key, stream);

            node_size = util_fread_int(stream);

            file_node_type file_node{status, node_offset, node_size};
            if (status == NODE_IN_USE) {
                file_node.data_size = util_fread_int(stream);
                file_node.data_offset = ftell(stream) - file_node.node_offset;
            }
            return file_node;
        } else {
            return file_node_type(NODE_INVALID, node_offset, 0);
        }
    }
    return std::nullopt;
}

/*
   This function will write the node information to file, this
   includes the NODE_END_TAG identifier which shoule be written to the
   end of the node.
*/

static void file_node_fwrite(const file_node_type &file_node, const char *key,
                             FILE *stream) {
    if (file_node.node_size == 0)
        util_abort("%s: trying to write node with z<ero size \n", __func__);
    {
        fseek__(stream, file_node.node_offset, SEEK_SET);
        util_fwrite_int(file_node.status, stream);
        if (file_node.status == NODE_IN_USE)
            util_fwrite_string(key, stream);
        util_fwrite_int(file_node.node_size, stream);
        util_fwrite_int(file_node.data_size, stream);
        fseek__(stream,
                file_node.node_offset + file_node.node_size -
                    sizeof NODE_END_TAG,
                SEEK_SET);
        util_fwrite_int(NODE_END_TAG, stream);
    }
}

/*
   This marks the start and end of the node with the integer tags:
   NODE_WRITE_ACTIVE_START and NODE_WRITE_ACTIVE_END, signalling this
   section in the data file is 'work in progress', and should be
   discarded if the application aborts during the write.

   When the write is complete file_node_fwrite() should be called,
   which will replace the NODE_WRITE_ACTIVE_START and
   NODE_WRITE_ACTIVE_END tags with NODE_IN_USE and NODE_END_TAG
   identifiers.
*/

static void file_node_init_fwrite(const file_node_type &file_node,
                                  FILE *stream) {
    fseek__(stream, file_node.node_offset, SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_START, stream);
    fseek__(stream,
            file_node.node_offset + file_node.node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_END, stream);
}

/*
   Observe that header in this context include the size of the tail
   marker NODE_END_TAG.
*/

static int file_node_header_size(const char *filename) {
    return sizeof(file_node_type::status) + sizeof(file_node_type::node_size) +
           sizeof(file_node_type::data_size) + sizeof(NODE_END_TAG) +
           sizeof(int) + strlen(filename) + 1;
}

static void file_node_set_data_offset(file_node_type &file_node,
                                      const char *filename) {
    file_node.data_offset =
        file_node_header_size(filename) - sizeof(NODE_END_TAG);
}

static file_node_type file_node_index_buffer_fread_alloc(buffer_type *buffer) {
    node_status_type status = (node_status_type)buffer_fread_int(buffer);
    long int node_offset = buffer_fread_long(buffer);
    int node_size = buffer_fread_int(buffer);
    {
        file_node_type file_node(status, node_offset, node_size);

        file_node.data_offset = buffer_fread_int(buffer);
        file_node.data_size = buffer_fread_int(buffer);

        return file_node;
    }
}

static block_fs_type *block_fs_alloc_empty(const fs::path &mount_file,
                                           bool read_only) {
    block_fs_type *block_fs = new block_fs_type;
    UTIL_TYPE_ID_INIT(block_fs, BLOCK_FS_TYPE_ID);

    block_fs->mount_path = mount_file.parent_path();
    block_fs->base_name = mount_file.stem();
    {
        FILE *stream = util_fopen(mount_file.c_str(), "r");
        int id = util_fread_int(stream);
        int version = util_fread_int(stream);
        if (version != 0)
            throw std::runtime_error(
                fmt::format("block_fs '{}' uses data version {} rather than 0",
                            mount_file, version));
        fclose(stream);

        if (id != MOUNT_MAP_MAGIC_INT)
            throw std::runtime_error(fmt::format(
                "The file '{}' is not a valid block_fs mount_map", mount_file));
    }

    block_fs->data_owner = !read_only;
    return block_fs;
}

UTIL_IS_INSTANCE_FUNCTION(block_fs, BLOCK_FS_TYPE_ID);

/*
   Will seek the datafile to the end of the current file_node. So that the next read will be "guaranteed" to
   start at a new node.
*/
static void block_fs_fseek_node_end(block_fs_type *block_fs,
                                    const file_node_type &file_node) {
    block_fs_fseek(block_fs, file_node.node_offset + file_node.node_size);
}

static void block_fs_fseek_node_data(block_fs_type *block_fs,
                                     const file_node_type &file_node) {
    block_fs_fseek(block_fs, file_node.node_offset + file_node.data_offset);
}

/*
   This function will read through the datafile seeking for one of the
   identifiers: NODE_IN_USE | NODE_FREE. If one of the valid status
   identifiers is found the stream is repositioned at the beginning of
   the valid node, so the calling scope can continue with a

      file_node = file_node_date_fread_alloc()

   call. If no valid status ID is found whatsover the data_stream
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
                /*
           OK - we found one interesting byte; let us try to read the
           whole integer and see if we have hit any of the valid status identifiers.
        */
                fseek__(block_fs->data_stream, -1, SEEK_CUR);
                if (fread(&status, sizeof status, 1, block_fs->data_stream) ==
                    1) {
                    if (status == NODE_IN_USE) {
                        /*
               OK - we have found a valid identifier. We reposition to
               the start of this valid status id and return true.
            */
                        fseek__(block_fs->data_stream, -sizeof status,
                                SEEK_CUR);
                        return true;
                    } else
                        /*
               OK - this was not a valid id; we go back and continue
               reading single bytes.
            */
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

/*
   This function will 'fix' the nodes with offset in offset_list.  The
   fixing in this case means the following:

     1. The node is updated in place on the file to become a free node.
     2. The node is added to the block_fs instance as a free node, which can
        be recycled at a later stage.

   If the instance is not data owner (i.e. read-only) the function
   will return immediately.
*/

static void block_fs_fix_nodes(block_fs_type *block_fs,
                               const std::vector<long> &offset_list) {
    if (block_fs->data_owner) {
        fsync(fileno(block_fs->data_stream));
        {
            char *key = NULL;
            for (const auto &node_offset : offset_list) {
                bool new_node = false;
                block_fs_fseek(block_fs, node_offset);
                auto file_node =
                    file_node_fread_alloc(block_fs->data_stream, &key).value();

                if ((file_node.status == NODE_INVALID) ||
                    (file_node.status == NODE_WRITE_ACTIVE)) {
                    /* This node is really quite broken. */
                    long int node_end;
                    block_fs_fseek_valid_node(block_fs);
                    node_end = ftell(block_fs->data_stream);
                    file_node.node_size = node_end - node_offset;
                }

                file_node.data_size = 0;
                file_node.data_offset = 0;

                block_fs_fseek(block_fs, node_offset);
                file_node_fwrite(file_node, NULL, block_fs->data_stream);
            }
            free(key);
        }
        fsync(fileno(block_fs->data_stream));
    }
}

static void block_fs_build_index(block_fs_type *block_fs,
                                 std::vector<long> &error_offset) {
    char *filename = NULL;

    block_fs_fseek(block_fs, 0);
    for (;;) {
        auto file_node_opt =
            file_node_fread_alloc(block_fs->data_stream, &filename);
        if (!file_node_opt.has_value())
            break;
        auto &file_node = *file_node_opt;
        if ((file_node.status == NODE_INVALID) ||
            (file_node.status == NODE_WRITE_ACTIVE)) {
            if (file_node.status == NODE_INVALID)
                fprintf(stderr,
                        "** Warning:: invalid node found at offset:%ld in "
                        "datafile:%s - data will be lost, node_size:%d\n",
                        file_node.node_offset, block_fs->data_path().c_str(),
                        file_node.node_size);
            else
                fprintf(stderr,
                        "** Warning:: file system was prematurely shut down "
                        "while writing node in %s/%ld - will be discarded.\n",
                        block_fs->data_path().c_str(), file_node.node_offset);

            error_offset.push_back(file_node.node_offset);
            block_fs_fseek_valid_node(block_fs);
        } else {
            if (file_node_verify_end_tag(file_node, block_fs->data_stream)) {
                block_fs_fseek_node_end(block_fs, file_node);
                switch (file_node.status) {
                case (NODE_IN_USE):
                    block_fs->index.emplace(filename, file_node);
                    break;
                default:
                    util_abort("%s: node status flag:%d not recognized - "
                               "error in data file \n",
                               __func__, file_node.status);
                }
            } else {
                /*
             Could not find a valid END_TAG - indicating that
             the filesystem was shut down during the write of
             this node.  This node will NOT be added to the
             index.  The node will be updated to become a free node.
          */
                fprintf(stderr,
                        "** Warning found node:%s at offset:%ld which was "
                        "incomplete - discarded.\n",
                        filename, file_node.node_offset);
                error_offset.push_back(file_node.node_offset);
                block_fs_fseek_valid_node(block_fs);
            }
        }
    };
    free(filename);
}

bool block_fs_is_readonly(const block_fs_type *bfs) {
    if (bfs->data_owner)
        return false;
    else
        return true;
}

block_fs_type *block_fs_mount(const char *mount_file, bool read_only) {
    block_fs_type *block_fs;
    {

        if (!fs::exists(mount_file)) {
            /* This is a brand new filesystem - create the mount map first. */
            FILE *stream = util_fopen(mount_file, "w");
            util_fwrite_int(MOUNT_MAP_MAGIC_INT, stream);
            util_fwrite_int(0 /* data version, always 0 */, stream);
            fclose(stream);
        }
        {
            std::vector<long> fix_nodes;
            block_fs = block_fs_alloc_empty(mount_file, read_only);

            block_fs->data_stream = fopen(block_fs->data_path().c_str(),
                                          block_fs->data_owner ? "ab+" : "rb");
            if (block_fs->data_stream)
                block_fs_build_index(block_fs, fix_nodes);
            block_fs_fix_nodes(block_fs, fix_nodes);
        }
    }
    return block_fs;
}

static file_node_type block_fs_get_new_node(block_fs_type *block_fs,
                                            const char *filename,
                                            size_t min_size) {
    long int offset;
    int node_size;

    // round min_size up to multiple of block_size
    const size_t block_size = 64;
    node_size = (min_size + block_size - 1) / block_size * block_size;

    /* Must lock the total size here ... */
    fseek(block_fs->data_stream, 0, SEEK_END);
    offset = ftell(block_fs->data_stream);
    return file_node_type(NODE_IN_USE, offset, node_size);
}

bool block_fs_has_file(block_fs_type *block_fs, const char *filename) {
    std::lock_guard guard{block_fs->mutex};
    bool has_file = block_fs->index.count(filename) > 0;
    return has_file;
}

/*
   It seems it is not enough to call fsync(); must also issue this
   funny fseek + ftell combination to ensure that all data is on
   disk after an uncontrolled shutdown.

   Could possibly use fdatasync() to improve speed slightly?
*/

void block_fs_fsync(block_fs_type *block_fs) {
    if (block_fs->data_owner) {
        //fdatasync( block_fs->data_fd );
        fsync(fileno(block_fs->data_stream));
        fseek(block_fs->data_stream, 0, SEEK_END);
        ftell(block_fs->data_stream);
    }
}

/*
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
                              file_node_type &node, const void *ptr,
                              int data_size) {
    block_fs_fseek(block_fs, node.node_offset);
    node.status = NODE_IN_USE;
    node.data_size = data_size;
    file_node_set_data_offset(node, filename);

    /* This marks the node section in the datafile as write in progress with: NODE_WRITE_ACTIVE_START ... NODE_WRITE_ACTIVE_END */
    file_node_init_fwrite(node, block_fs->data_stream);

    /* Writes the actual data content. */
    block_fs_fseek_node_data(block_fs, node);
    util_fwrite(ptr, 1, data_size, block_fs->data_stream, __func__);

    /* Writes the file node header data, including the NODE_END_TAG. */
    file_node_fwrite(node, filename, block_fs->data_stream);
}

static void block_fs_fwrite_file_unlocked(block_fs_type *block_fs,
                                          const char *filename, const void *ptr,
                                          size_t data_size) {}

void block_fs_fwrite_file(block_fs_type *block_fs, const char *filename,
                          const void *ptr, size_t data_size) {
    if (!block_fs->data_owner)
        throw std::logic_error(fmt::format(
            "tried to write to read only filesystem mouunted at: {}",
            block_fs->mount_path / block_fs->base_name));

    std::lock_guard guard{block_fs->mutex};

    bool new_node = true;
    size_t min_size = data_size + file_node_header_size(filename);

    if (block_fs->index.count(filename) > 0) {
        auto file_node = block_fs->index[filename];
        if (file_node.node_size < min_size) {
            file_node = block_fs_get_new_node(block_fs, filename, min_size);
        } else
            new_node = false; /* We are reusing the existing node. */
        block_fs_fwrite__(block_fs, filename, file_node, ptr, data_size);
        if (new_node)
            block_fs->index.emplace(filename, file_node);
    } else {
        auto file_node = block_fs_get_new_node(block_fs, filename, min_size);
        block_fs_fwrite__(block_fs, filename, file_node, ptr, data_size);
        if (new_node)
            block_fs->index.emplace(filename, file_node);
    }
}

/*
   Reads the full content of 'filename' into the buffer.
*/

void block_fs_fread_realloc_buffer(block_fs_type *block_fs,
                                   const char *filename, buffer_type *buffer) {
    std::lock_guard guard{block_fs->mutex};
    {
        auto node = block_fs->index[filename];

        buffer_clear(buffer); /* Setting: content_size = 0; pos = 0;  */

        block_fs_fseek_node_data(block_fs, node);
        buffer_stream_fread(buffer, node.data_size, block_fs->data_stream);
        //file_node_verify_end_tag( node , block_fs->data_stream );

        buffer_rewind(buffer); /* Setting: pos = 0; */
    }
}

/*
   Close/synchronize the open file descriptors and free all memory
   related to the block_fs instance.

   If the boolean unlink_empty is set to true all the files will be
   unlinked if the filesystem is empty.
*/

void block_fs_close(block_fs_type *block_fs) {
    block_fs_fsync(block_fs);

    if (block_fs->data_owner)
        block_fs->mutex.lock();

    if (block_fs->data_stream != NULL)
        fclose(block_fs->data_stream);

    if (block_fs->lock_fd > 0) {
        close(
            block_fs
                ->lock_fd); /* Closing the lock_file file descriptor - and releasing the lock. */
        util_unlink_existing(block_fs->lock_path().c_str());
    }

    if (block_fs->data_owner) {
        block_fs->mutex.unlock();
    }

    delete block_fs;
}
