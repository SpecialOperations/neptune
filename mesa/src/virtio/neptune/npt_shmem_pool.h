/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Growable shmem bump-allocator pool.  Each pool_alloc returns a
 * refcounted shmem + offset; on grow, in-flight callers' refs keep
 * the old shmem alive until released.
 */

#ifndef NPT_SHMEM_POOL_H
#define NPT_SHMEM_POOL_H

#include "npt_common.h"
#include "npt_renderer.h"
#include "util/list.h"

struct npt_shmem_pool {
   mtx_t mutex;
   size_t min_alloc_size;
   bool reusable;
   /* Bump-only mode (reusable == false). */
   struct npt_renderer_shmem *shmem;
   size_t size;
   size_t used;
   /* True when shmem was just created and no caller has yet
    * roundtripped to ensure the host's RESOURCE_CREATE_BLOB has
    * landed. */
   bool shmem_needs_roundtrip;
   /* Reusable slab mode (reusable == true): list of struct npt_shmem_slab
    * with per-slab coalescing free-lists and empty-slab reclamation. */
   struct list_head slabs;
   uint32_t empty_slabs;
};

void
npt_shmem_pool_init(struct npt_shmem_pool *pool, size_t min_alloc_size);

/* Initialize a pool that supports returning freed ranges via
 * npt_shmem_pool_free(), reusing slots across allocations, giving oversized
 * allocations dedicated shmems, and reclaiming empty slabs. */
void
npt_shmem_pool_init_reusable(struct npt_shmem_pool *pool, size_t slab_size);

void
npt_shmem_pool_fini(struct npt_renderer *renderer,
                    struct npt_shmem_pool *pool);

/*
 * Caller takes the returned shmem ref and must unref (or call
 * npt_shmem_pool_free for reusable pools) when done with the region.
 * *out_shmem_fresh=true => shmem was just created; caller must roundtrip
 * before naming res_id over the ring.  NULL out_shmem_fresh is fine if the
 * shmem is never handed to the host.
 */
struct npt_renderer_shmem *
npt_shmem_pool_alloc(struct npt_renderer *renderer,
                     struct npt_shmem_pool *pool,
                     size_t size, size_t *out_offset,
                     bool *out_shmem_fresh);

/*
 * Return a sub-allocated region (or dedicated oversized shmem) obtained from
 * a reusable pool and asynchronously release the caller's shmem reference.
 * Caller must ensure the host has finished reading [offset, offset + size).
 */
void
npt_shmem_pool_free(struct npt_renderer *renderer,
                    struct npt_shmem_pool *pool,
                    struct npt_renderer_shmem *shmem,
                    size_t offset, size_t size);

#endif /* NPT_SHMEM_POOL_H */
