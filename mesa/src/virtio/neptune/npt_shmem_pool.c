/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 */

#include "npt_shmem_pool.h"

#include <stdlib.h>

#define NPT_SHMEM_POOL_REUSABLE_ALIGN 64u
/* Keep at most one empty slab warm to avoid RESOURCE_CREATE_BLOB churn on
 * steady-state frames while still returning excess slabs to the OS. */
#define NPT_SHMEM_POOL_MAX_EMPTY_SLABS 1u

struct npt_shmem_free_range {
   size_t offset;
   size_t size;
   struct npt_shmem_free_range *next;
};

struct npt_shmem_slab {
   struct list_head head;
   struct npt_renderer_shmem *shmem;
   size_t size;
   size_t used_bytes;
   uint32_t active_allocs;
   bool needs_roundtrip;
   struct npt_shmem_free_range *free_list;
};

static inline size_t
align_up_reusable(size_t size)
{
   if (!size)
      size = NPT_SHMEM_POOL_REUSABLE_ALIGN;
   return (size + (NPT_SHMEM_POOL_REUSABLE_ALIGN - 1u)) &
          ~(size_t)(NPT_SHMEM_POOL_REUSABLE_ALIGN - 1u);
}

static void
slab_free_ranges_destroy(struct npt_shmem_slab *slab)
{
   struct npt_shmem_free_range *r = slab->free_list;
   while (r) {
      struct npt_shmem_free_range *next = r->next;
      free(r);
      r = next;
   }
   slab->free_list = NULL;
}

static bool
slab_alloc_range(struct npt_shmem_slab *slab, size_t size, size_t *out_offset)
{
   struct npt_shmem_free_range **link = &slab->free_list;
   while (*link) {
      struct npt_shmem_free_range *r = *link;
      if (r->size >= size) {
         *out_offset = r->offset;
         if (r->size == size) {
            *link = r->next;
            free(r);
         } else {
            r->offset += size;
            r->size -= size;
         }
         slab->active_allocs++;
         slab->used_bytes += size;
         return true;
      }
      link = &r->next;
   }
   return false;
}

static bool
slab_has_fit(const struct npt_shmem_slab *slab, size_t size)
{
   if (slab->size - slab->used_bytes < size)
      return false;
   for (const struct npt_shmem_free_range *r = slab->free_list; r; r = r->next) {
      if (r->size >= size)
         return true;
   }
   return false;
}

static void
slab_release_range(struct npt_shmem_slab *slab, size_t offset, size_t size)
{
   struct npt_shmem_free_range *prev = NULL;
   struct npt_shmem_free_range *cur = slab->free_list;
   while (cur && cur->offset < offset) {
      prev = cur;
      cur = cur->next;
   }

   const bool merge_prev = prev && (prev->offset + prev->size == offset);
   const bool merge_next = cur && (offset + size == cur->offset);

   if (merge_prev && merge_next) {
      prev->size += size + cur->size;
      prev->next = cur->next;
      free(cur);
   } else if (merge_prev) {
      prev->size += size;
   } else if (merge_next) {
      cur->offset = offset;
      cur->size += size;
   } else {
      struct npt_shmem_free_range *node = malloc(sizeof(*node));
      if (node) {
         node->offset = offset;
         node->size = size;
         node->next = cur;
         if (prev)
            prev->next = node;
         else
            slab->free_list = node;
      }
   }

   if (slab->active_allocs > 0)
      slab->active_allocs--;
   if (slab->used_bytes >= size)
      slab->used_bytes -= size;
   else
      slab->used_bytes = 0;
}

void
npt_shmem_pool_init(struct npt_shmem_pool *pool, size_t min_alloc_size)
{
   memset(pool, 0, sizeof(*pool));
   pool->min_alloc_size = 1;
   while (pool->min_alloc_size < min_alloc_size)
      pool->min_alloc_size <<= 1;
   list_inithead(&pool->slabs);
   mtx_init(&pool->mutex, mtx_plain);
}

void
npt_shmem_pool_init_reusable(struct npt_shmem_pool *pool, size_t slab_size)
{
   npt_shmem_pool_init(pool, slab_size);
   pool->reusable = true;
}

void
npt_shmem_pool_fini(struct npt_renderer *renderer,
                    struct npt_shmem_pool *pool)
{
   if (pool->reusable) {
      list_for_each_entry_safe (struct npt_shmem_slab, slab, &pool->slabs, head) {
         list_del(&slab->head);
         slab_free_ranges_destroy(slab);
         if (slab->shmem)
            npt_renderer_shmem_unref(renderer, slab->shmem);
         free(slab);
      }
      pool->empty_slabs = 0;
   } else if (pool->shmem) {
      npt_renderer_shmem_unref(renderer, pool->shmem);
      pool->shmem = NULL;
   }
   mtx_destroy(&pool->mutex);
}

static bool
npt_shmem_pool_grow_locked(struct npt_renderer *renderer,
                           struct npt_shmem_pool *pool,
                           size_t size)
{
   size_t alloc_size = pool->min_alloc_size;
   while (alloc_size < size) {
      alloc_size <<= 1;
      if (!alloc_size)
         return false;
   }

   struct npt_renderer_shmem *shmem =
      npt_renderer_shmem_create(renderer, alloc_size);
   if (!shmem) {
      npt_log("shmem pool: failed to allocate %zu bytes", alloc_size);
      return false;
   }

   if (pool->shmem)
      npt_renderer_shmem_unref(renderer, pool->shmem);

   pool->shmem = shmem;
   pool->size = alloc_size;
   pool->used = 0;
   /* RESOURCE_CREATE_BLOB is async vs the ring thread; flag so the
    * first caller pays the virtqueue-seqno roundtrip before naming
    * res_id over the ring. */
   pool->shmem_needs_roundtrip = true;
   return true;
}

static struct npt_renderer_shmem *
npt_shmem_pool_alloc_reusable(struct npt_renderer *renderer,
                              struct npt_shmem_pool *pool,
                              size_t size, size_t *out_offset,
                              bool *out_shmem_fresh)
{
   const size_t aligned = align_up_reusable(size);

   /* Oversized allocations get their own dedicated shmem so a single huge
    * staging texture never inflates a shared slab or gets pinned by small
    * long-lived buffers. */
   if (aligned > pool->min_alloc_size / 2u) {
      struct npt_renderer_shmem *dedicated =
         npt_renderer_shmem_create(renderer, aligned);
      if (!dedicated) {
         npt_log("shmem pool: failed dedicated alloc of %zu bytes", aligned);
         if (out_shmem_fresh)
            *out_shmem_fresh = false;
         return NULL;
      }
      *out_offset = 0;
      if (out_shmem_fresh)
         *out_shmem_fresh = true;
      return dedicated;
   }

   mtx_lock(&pool->mutex);

   /* Pack into the most-occupied non-empty slab that fits so sparsely used
    * slabs naturally drain to zero and get reclaimed. */
   struct npt_shmem_slab *best = NULL;
   list_for_each_entry (struct npt_shmem_slab, slab, &pool->slabs, head) {
      if (!slab_has_fit(slab, aligned))
         continue;
      if (!best) {
         best = slab;
      } else if (best->active_allocs == 0 && slab->active_allocs > 0) {
         best = slab;
      } else if (slab->active_allocs > 0 && best->active_allocs > 0 &&
                 slab->used_bytes > best->used_bytes) {
         best = slab;
      }
   }

   if (best) {
      const bool was_empty = (best->active_allocs == 0);
      size_t off = 0;
      if (slab_alloc_range(best, aligned, &off)) {
         if (was_empty && pool->empty_slabs > 0)
            pool->empty_slabs--;
         struct npt_renderer_shmem *shmem =
            npt_renderer_shmem_ref(renderer, best->shmem);
         *out_offset = off;
         if (out_shmem_fresh) {
            *out_shmem_fresh = best->needs_roundtrip;
            best->needs_roundtrip = false;
         }
         mtx_unlock(&pool->mutex);
         return shmem;
      }
   }

   mtx_unlock(&pool->mutex);

   /* Allocate a fresh slab outside pool->mutex so shmem creation doesn't
    * stall concurrent frees or allocations. */
   const size_t slab_size = pool->min_alloc_size;
   struct npt_renderer_shmem *new_shmem =
      npt_renderer_shmem_create(renderer, slab_size);
   if (!new_shmem) {
      npt_log("shmem pool: failed to allocate %zu byte slab", slab_size);
      if (out_shmem_fresh)
         *out_shmem_fresh = false;
      return NULL;
   }

   struct npt_shmem_slab *slab = calloc(1, sizeof(*slab));
   struct npt_shmem_free_range *rem = NULL;
   if (slab_size > aligned)
      rem = malloc(sizeof(*rem));
   if (!slab || (slab_size > aligned && !rem)) {
      free(rem);
      free(slab);
      npt_renderer_shmem_unref(renderer, new_shmem);
      if (out_shmem_fresh)
         *out_shmem_fresh = false;
      return NULL;
   }

   slab->shmem = new_shmem;
   slab->size = slab_size;
   slab->used_bytes = aligned;
   slab->active_allocs = 1;
   slab->needs_roundtrip = false;
   if (rem) {
      rem->offset = aligned;
      rem->size = slab_size - aligned;
      rem->next = NULL;
      slab->free_list = rem;
   }

   /* Caller gets its own ref in addition to the slab's pool ref. */
   struct npt_renderer_shmem *caller_ref =
      npt_renderer_shmem_ref(renderer, new_shmem);

   mtx_lock(&pool->mutex);
   list_add(&slab->head, &pool->slabs);
   mtx_unlock(&pool->mutex);

   *out_offset = 0;
   if (out_shmem_fresh)
      *out_shmem_fresh = true;
   return caller_ref;
}

struct npt_renderer_shmem *
npt_shmem_pool_alloc(struct npt_renderer *renderer,
                     struct npt_shmem_pool *pool,
                     size_t size, size_t *out_offset,
                     bool *out_shmem_fresh)
{
   if (pool->reusable) {
      return npt_shmem_pool_alloc_reusable(renderer, pool, size, out_offset,
                                           out_shmem_fresh);
   }

   mtx_lock(&pool->mutex);

   if (size > pool->size - pool->used) {
      if (!npt_shmem_pool_grow_locked(renderer, pool, size)) {
         mtx_unlock(&pool->mutex);
         if (out_shmem_fresh)
            *out_shmem_fresh = false;
         return NULL;
      }
   }

   struct npt_renderer_shmem *shmem =
      npt_renderer_shmem_ref(renderer, pool->shmem);
   *out_offset = pool->used;
   pool->used += size;
   pool->used = (pool->used + 7) & ~(size_t)7;

   /* Report-once: only the first alloc after a grow pays the roundtrip. */
   if (out_shmem_fresh) {
      *out_shmem_fresh = pool->shmem_needs_roundtrip;
      pool->shmem_needs_roundtrip = false;
   }

   mtx_unlock(&pool->mutex);
   return shmem;
}

void
npt_shmem_pool_free(struct npt_renderer *renderer,
                    struct npt_shmem_pool *pool,
                    struct npt_renderer_shmem *shmem,
                    size_t offset, size_t size)
{
   if (!shmem)
      return;

   struct npt_renderer_shmem *drop_pool_ref = NULL;

   if (pool && pool->reusable) {
      const size_t aligned = align_up_reusable(size);
      mtx_lock(&pool->mutex);
      list_for_each_entry (struct npt_shmem_slab, slab, &pool->slabs, head) {
         if (slab->shmem == shmem) {
            slab_release_range(slab, offset, aligned);
            if (slab->active_allocs == 0) {
               if (pool->empty_slabs >= NPT_SHMEM_POOL_MAX_EMPTY_SLABS) {
                  list_del(&slab->head);
                  slab_free_ranges_destroy(slab);
                  drop_pool_ref = slab->shmem;
                  free(slab);
               } else {
                  pool->empty_slabs++;
                  /* Move empty warm slab to the tail. */
                  list_del(&slab->head);
                  list_addtail(&slab->head, &pool->slabs);
               }
            }
            break;
         }
      }
      mtx_unlock(&pool->mutex);
   }

   npt_renderer_shmem_unref_async(renderer, shmem);
   if (drop_pool_ref)
      npt_renderer_shmem_unref_async(renderer, drop_pool_ref);
}
