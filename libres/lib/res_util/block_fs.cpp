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

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <errno.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fmt/ostream.h>

#include <ert/python.hpp>

#include <ert/res_util/block_fs.hpp>

namespace fs = std::filesystem;

#define MOUNT_MAP_MAGIC_INT 8861290
#define BLOCK_FS_TYPE_ID 7100652

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

/**
 * Meta-information about where to find the data for a block.
 */
struct BlockMeta {
    int64_t node_offset{};
    int32_t data_offset{};
    int32_t node_size{};
    int32_t data_size{};
    node_status_type status = NODE_INVALID;

    BlockMeta() = default;
    BlockMeta(const BlockMeta &) = default;
    BlockMeta(BlockMeta &&) = default;

    BlockMeta(node_status_type status, int64_t offset, int32_t size)
        : node_offset(offset), node_size(size), data_size(size),
          status(status) {}

    BlockMeta &operator=(const BlockMeta &) = default;
    BlockMeta &operator=(BlockMeta &&) = default;
};

struct block_fs_struct {
    UTIL_TYPE_ID_DECLARATION;

    int data_fd;
    FILE *data_stream;

    std::mutex mutex;

    std::unordered_map<std::string, BlockMeta> index;
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

static bool file_node_verify_end_tag(const BlockMeta &meta, FILE *stream) {
    int end_tag;
    fseek__(stream, meta.node_offset + meta.node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    if (fread(&end_tag, sizeof end_tag, 1, stream) == 1) {
        if (end_tag == NODE_END_TAG)
            return true;
        else
            return false;
    } else
        return false;
}

static std::optional<BlockMeta> file_node_fread_alloc(FILE *stream,
                                                      char **key) {
    node_status_type status;
    long int node_offset = ftell(stream);
    if (fread(&status, sizeof status, 1, stream) == 1) {
        if (status == NODE_IN_USE) {
            int node_size;
            *key = util_fread_realloc_string(*key, stream);

            node_size = util_fread_int(stream);
            if (node_size <= 0)
                status = NODE_INVALID;
            BlockMeta block{status, node_offset, node_size};
            block.data_size = util_fread_int(stream);
            block.data_offset = ftell(stream) - block.node_offset;
            return block;
        } else {
            if (status != NODE_WRITE_ACTIVE)
                status = NODE_INVALID;
            return BlockMeta{status, node_offset, 0};
        }
    }
    return std::nullopt;
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
static void file_node_fwrite(const BlockMeta &block, const char *key,
                             FILE *stream) {
    if (block.node_size == 0)
        util_abort("%s: trying to write node with z<ero size \n", __func__);
    {
        fseek__(stream, block.node_offset, SEEK_SET);
        util_fwrite_int(block.status, stream);
        if (block.status == NODE_IN_USE)
            util_fwrite_string(key, stream);
        util_fwrite_int(block.node_size, stream);
        util_fwrite_int(block.data_size, stream);
        fseek__(stream,
                block.node_offset + block.node_size - sizeof NODE_END_TAG,
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
static void file_node_init_fwrite(const BlockMeta &block, FILE *stream) {
    fseek__(stream, block.node_offset, SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_START, stream);
    fseek__(stream, block.node_offset + block.node_size - sizeof NODE_END_TAG,
            SEEK_SET);
    util_fwrite_int(NODE_WRITE_ACTIVE_END, stream);
}

/**
   Observe that header in this context includes the size of the tail
   marker NODE_END_TAG.
*/
static int file_node_header_size(const char *filename) {
    return sizeof(BlockMeta::status) + sizeof(BlockMeta::node_size) +
           sizeof(BlockMeta::data_size) + sizeof(NODE_END_TAG) +
           sizeof(int) /* embedded by the util_fwrite_string routine */ +
           strlen(filename) + 1 /* \0 */;
}

static void file_node_set_data_offset(BlockMeta &block, const char *filename) {
    block.data_offset = file_node_header_size(filename) - sizeof(NODE_END_TAG);
}

static BlockMeta file_node_index_buffer_fread_alloc(buffer_type *buffer) {
    node_status_type status = (node_status_type)buffer_fread_int(buffer);
    long int node_offset = buffer_fread_long(buffer);
    int node_size = buffer_fread_int(buffer);

    BlockMeta block{status, node_offset, node_size};
    block.data_offset = buffer_fread_int(buffer);
    block.data_size = buffer_fread_int(buffer);
    return block;
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
                                    const BlockMeta &block) {
    block_fs_fseek(block_fs, block.node_offset + block.node_size);
}

static void block_fs_fseek_node_data(block_fs_type *block_fs,
                                     const BlockMeta &block) {
    block_fs_fseek(block_fs, block.node_offset + block.data_offset);
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
            block_fs_fseek(block_fs, node_offset);
            auto block = file_node_fread_alloc(block_fs->data_stream, &key);

            if ((block->status == NODE_INVALID) ||
                (block->status == NODE_WRITE_ACTIVE)) {
                /* This node is really quite broken. */
                long int node_end;
                block_fs_fseek_valid_node(block_fs);
                node_end = ftell(block_fs->data_stream);
                block->node_size = node_end - node_offset;
            }

            block->status = NODE_INVALID;
            block->data_size = 0;
            block->data_offset = 0;

            block_fs_fseek(block_fs, node_offset);
            file_node_fwrite(*block, NULL, block_fs->data_stream);
        }
        free(key);

        fsync(block_fs->data_fd);
    }
}

static void block_fs_build_index(block_fs_type *block_fs,
                                 const fs::path &data_file,
                                 std::vector<long> &error_offset) {
    char *filename = NULL;

    block_fs_fseek(block_fs, 0);
    for (;;) {
        auto block = file_node_fread_alloc(block_fs->data_stream, &filename);
        if (!block.has_value())
            break;

        if ((block->status == NODE_INVALID) ||
            (block->status == NODE_WRITE_ACTIVE)) {
            if (block->status == NODE_INVALID)
                fprintf(stderr,
                        "** Warning:: invalid node found at offset:%ld in "
                        "datafile:%s - data will be lost, node_size:%d\n",
                        block->node_offset, data_file.c_str(),
                        block->node_size);
            else
                fprintf(stderr,
                        "** Warning:: file system was prematurely shut down "
                        "while writing node in %s/%ld - will be discarded.\n",
                        data_file.c_str(), block->node_offset);

            error_offset.push_back(block->node_offset);
            block_fs_fseek_valid_node(block_fs);
        } else {
            if (file_node_verify_end_tag(*block, block_fs->data_stream)) {
                block_fs_fseek_node_end(block_fs, *block);
                if (block->status == NODE_IN_USE) {
                    block_fs->index[filename] = *block;
                } else {
                    util_abort("%s: node status flag:%d not recognized - "
                               "error in data file \n",
                               __func__, block->status);
                }
            } else {
                // Could not find a valid END_TAG - indicating that
                // the filesystem was shut down during the write of
                // this node.  This node will NOT be added to the
                // index.  The node will be updated to become a free node.
                fprintf(stderr,
                        "** Warning found node:%s at offset:%ld which was "
                        "incomplete - discarded.\n",
                        filename, block->node_offset);
                error_offset.push_back(block->node_offset);
                block_fs_fseek_valid_node(block_fs);
            }
        }
    }
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

static BlockMeta block_fs_get_new_node(block_fs_type *block_fs,
                                       const char *filename, size_t size) {
    /* Must lock the total size here ... */
    fseek(block_fs->data_stream, 0, SEEK_END);
    int64_t offset = ftell(block_fs->data_stream);
    BlockMeta block{NODE_IN_USE, offset, static_cast<int32_t>(size)};
    block_fs->index[filename] = block;

    return block;
}

bool block_fs_has_file(block_fs_type *block_fs, const char *filename) {
    std::lock_guard guard{block_fs->mutex};
    return block_fs->index.count(filename) > 0;
}

void block_fs_fsync(block_fs_type *block_fs) {
    if (!block_fs_is_readonly(block_fs)) {
        fsync(block_fs->data_fd);
        fseek__(block_fs->data_stream, 0, SEEK_END);
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
                              BlockMeta &block, const void *ptr,
                              int data_size) {
    block_fs_fseek(block_fs, block.node_offset);
    block.status = NODE_IN_USE;
    block.data_size = data_size;
    file_node_set_data_offset(block, filename);

    // This marks the node section in the datafile as write in progress with:
    // NODE_WRITE_ACTIVE_START ... NODE_WRITE_ACTIVE_END
    file_node_init_fwrite(block, block_fs->data_stream);

    /* Writes the actual data content. */
    block_fs_fseek_node_data(block_fs, block);
    util_fwrite(ptr, 1, data_size, block_fs->data_stream, __func__);

    /* Writes the file node header data, including the NODE_END_TAG. */
    file_node_fwrite(block, filename, block_fs->data_stream);
}

void block_fs_fwrite_file(block_fs_type *block_fs, const char *filename,
                          const void *ptr, size_t data_size) {
    if (block_fs_is_readonly(block_fs))
        throw std::runtime_error("tried to write to read only filesystem");
    std::lock_guard guard{block_fs->mutex};

    size_t min_size = data_size + file_node_header_size(filename);

    BlockMeta block = block_fs_get_new_node(block_fs, filename, min_size);

    /* The actual writing ... */
    block_fs_fwrite__(block_fs, filename, block, ptr, data_size);
    block_fs->index[filename] = block;
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
    const BlockMeta &block = block_fs->index.at(filename);

    buffer_clear(buffer); /* Setting: content_size = 0; pos = 0;  */

    block_fs_fseek_node_data(block_fs, block);
    buffer_stream_fread(buffer, block.data_size, block_fs->data_stream);

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

    delete block_fs;
}
